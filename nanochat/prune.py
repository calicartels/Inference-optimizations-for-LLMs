import torch
from nanochat.gpt import GPT, GPTConfig


def head_imp(model):
    head_importance = {}
    head_dim = model.config.n_embd // model.config.n_head
    
    for layer_idx, block in enumerate(model.transformer.h):
        attn = block.attn
        n_head = attn.n_head
        n_kv_head = attn.n_kv_head
        
        head_scores = torch.zeros(n_head, device=next(model.parameters()).device)
        
        q_weight = attn.c_q.weight.view(n_head, head_dim, model.config.n_embd)
        head_scores += q_weight.abs().sum(dim=(1, 2))
        
        proj_weight = attn.c_proj.weight.view(model.config.n_embd, n_head, head_dim)
        head_scores += proj_weight.abs().sum(dim=(0, 2))
        
        if n_kv_head == n_head:
            k_weight = attn.c_k.weight.view(n_head, head_dim, model.config.n_embd)
            v_weight = attn.c_v.weight.view(n_head, head_dim, model.config.n_embd)
            head_scores += k_weight.abs().sum(dim=(1, 2))
            head_scores += v_weight.abs().sum(dim=(1, 2))
        
        head_importance[layer_idx] = head_scores
    
    return head_importance


def neuron_imp(model):
    neuron_importance = {}
    hidden_dim = 4 * model.config.n_embd
    
    for layer_idx, block in enumerate(model.transformer.h):
        mlp = block.mlp
        
        fc_importance = mlp.c_fc.weight.abs().sum(dim=1)
        proj_importance = mlp.c_proj.weight.abs().sum(dim=0)
        
        neuron_scores = fc_importance + proj_importance
        neuron_importance[layer_idx] = neuron_scores
    
    return neuron_importance


def select_heads(head_importance, prune_ratio):
    heads_to_keep = {}
    
    for layer_idx, scores in head_importance.items():
        n_head = len(scores)
        n_to_keep = int(n_head * (1 - prune_ratio))
        n_to_keep = max(1, n_to_keep)
        
        _, top_indices = torch.topk(scores, n_to_keep, largest=True)
        heads_to_keep[layer_idx] = top_indices.sort().values.tolist()
    
    return heads_to_keep


def select_neurons(neuron_importance, prune_ratio):
    neurons_to_keep = {}
    
    for layer_idx, scores in neuron_importance.items():
        n_neurons = len(scores)
        n_to_keep = int(n_neurons * (1 - prune_ratio))
        n_to_keep = max(1, n_to_keep)
        
        _, top_indices = torch.topk(scores, n_to_keep, largest=True)
        neurons_to_keep[layer_idx] = top_indices.sort().values.tolist()
    
    return neurons_to_keep


def make_pruned_config(original_config, heads_to_keep, neurons_to_keep):
    min_heads = min(len(heads) for heads in heads_to_keep.values())
    min_neurons = min(len(neurons) for neurons in neurons_to_keep.values())
    
    head_dim = original_config.n_embd // original_config.n_head
    new_n_embd = min_heads * head_dim
    
    config = GPTConfig(
        sequence_len=original_config.sequence_len,
        vocab_size=original_config.vocab_size,
        n_layer=original_config.n_layer,
        n_head=min_heads,
        n_kv_head=1 if original_config.use_mqa else min_heads,
        n_embd=new_n_embd,
        window_pattern=original_config.window_pattern,
        use_mqa=original_config.use_mqa,
        multi_token_n=original_config.multi_token_n,
        draft_n=original_config.draft_n,
        draft_hidden_mult=original_config.draft_hidden_mult,
    )
    
    return config, min_heads, min_neurons


def prune_weights(model, heads_to_keep, neurons_to_keep, pruned_config, min_heads, min_neurons):
    head_dim = model.config.n_embd // model.config.n_head
    original_n_embd = model.config.n_embd
    pruned_n_embd = pruned_config.n_embd
    
    with torch.device("meta"):
        pruned_model = GPT(pruned_config)
    
    device = next(model.parameters()).device
    pruned_model.to_empty(device=device)
    
    pruned_model.transformer.wte.weight.data.copy_(model.transformer.wte.weight.data[:, :pruned_n_embd])
    
    pruned_model.resid_lambdas.data.copy_(model.resid_lambdas.data)
    pruned_model.x0_lambdas.data.copy_(model.x0_lambdas.data)
    
    for key in model.value_embeds.keys():
        if key in pruned_model.value_embeds:
            orig_ve = model.value_embeds[key].weight.data
            pruned_ve = pruned_model.value_embeds[key].weight.data
            if orig_ve.size(1) > pruned_ve.size(1):
                pruned_model.value_embeds[key].weight.data.copy_(orig_ve[:, :pruned_ve.size(1)])
            else:
                pruned_model.value_embeds[key].weight.data.copy_(orig_ve)
    
    for key in model.multi_token_heads.keys():
        if key in pruned_model.multi_token_heads:
            orig_weight = model.multi_token_heads[key].weight.data
            pruned_model.multi_token_heads[key].weight.data.copy_(orig_weight[:, :pruned_n_embd])
    
    if model.draft_head is not None and pruned_model.draft_head is not None:
        pruned_model.draft_head.fc1.weight.data.copy_(model.draft_head.fc1.weight.data[:, :pruned_n_embd])
        pruned_model.draft_head.fc2.weight.data.copy_(model.draft_head.fc2.weight.data)
    
    pruned_model.lm_head.weight.data.copy_(model.lm_head.weight.data[:, :pruned_n_embd])
    
    for layer_idx in range(model.config.n_layer):
        orig_block = model.transformer.h[layer_idx]
        pruned_block = pruned_model.transformer.h[layer_idx]
        
        layer_heads = heads_to_keep[layer_idx]
        layer_neurons = neurons_to_keep[layer_idx]
        
        attn_orig = orig_block.attn
        attn_pruned = pruned_block.attn
        
        q_orig = attn_orig.c_q.weight.view(model.config.n_head, head_dim, original_n_embd)
        q_pruned = q_orig[layer_heads[:min_heads]].contiguous().view(min_heads * head_dim, pruned_n_embd)
        attn_pruned.c_q.weight.data.copy_(q_pruned)
        
        if attn_orig.n_kv_head == model.config.n_head:
            k_orig = attn_orig.c_k.weight.view(model.config.n_head, head_dim, original_n_embd)
            k_pruned = k_orig[layer_heads[:min_heads]].contiguous().view(min_heads * head_dim, pruned_n_embd)
            attn_pruned.c_k.weight.data.copy_(k_pruned)
        else:
            k_orig = attn_orig.c_k.weight
            k_pruned = k_orig[:, :pruned_n_embd]
            attn_pruned.c_k.weight.data.copy_(k_pruned)
        
        if attn_orig.n_kv_head == model.config.n_head:
            v_orig = attn_orig.c_v.weight.view(model.config.n_head, head_dim, original_n_embd)
            v_pruned = v_orig[layer_heads[:min_heads]].contiguous().view(min_heads * head_dim, pruned_n_embd)
            attn_pruned.c_v.weight.data.copy_(v_pruned)
        else:
            v_orig = attn_orig.c_v.weight
            v_pruned = v_orig[:, :pruned_n_embd]
            attn_pruned.c_v.weight.data.copy_(v_pruned)
        
        proj_orig = attn_orig.c_proj.weight.view(original_n_embd, model.config.n_head, head_dim)
        proj_pruned = proj_orig[:, layer_heads[:min_heads], :].contiguous().view(original_n_embd, min_heads * head_dim)
        proj_pruned = proj_pruned[:pruned_n_embd, :]
        attn_pruned.c_proj.weight.data.copy_(proj_pruned)
        
        if attn_orig.ve_gate is not None and attn_pruned.ve_gate is not None:
            if attn_orig.n_kv_head == model.config.n_head:
                gate_orig = attn_orig.ve_gate.weight.view(model.config.n_head, -1)
                gate_pruned = gate_orig[layer_heads[:min_heads]]
                attn_pruned.ve_gate.weight.data.copy_(gate_pruned.view(min_heads, -1))
            else:
                attn_pruned.ve_gate.weight.data.copy_(attn_orig.ve_gate.weight.data)
        
        mlp_orig = orig_block.mlp
        mlp_pruned = pruned_block.mlp
        
        fc_orig = mlp_orig.c_fc.weight
        fc_pruned = fc_orig[layer_neurons[:min_neurons]]
        fc_pruned = fc_pruned[:, :pruned_n_embd]
        mlp_pruned.c_fc.weight.data.copy_(fc_pruned)
        
        proj_orig = mlp_orig.c_proj.weight
        proj_pruned = proj_orig[:, layer_neurons[:min_neurons]]
        proj_pruned = proj_pruned[:pruned_n_embd, :]
        mlp_pruned.c_proj.weight.data.copy_(proj_pruned)
    
    pruned_model.cos.copy_(model.cos)
    pruned_model.sin.copy_(model.sin)
    
    return pruned_model


def prune_model(model, head_prune_ratio=0.2, neuron_prune_ratio=0.2):
    head_importance = head_imp(model)
    neuron_importance = neuron_imp(model)
    
    heads_to_keep = select_heads(head_importance, head_prune_ratio)
    neurons_to_keep = select_neurons(neuron_importance, neuron_prune_ratio)
    
    config, min_heads, min_neurons = make_pruned_config(
        model.config, heads_to_keep, neurons_to_keep
    )
    
    pruned_model = prune_weights(
        model, heads_to_keep, neurons_to_keep, config, min_heads, min_neurons
    )
    
    return pruned_model, config
