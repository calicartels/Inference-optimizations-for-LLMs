import torch
import torch.nn.functional as F


def compute_distillation_loss(
    student_logits,
    teacher_logits,
    temperature=1.0,
    reduction='mean'
):
    """
    Compute KL divergence loss between student and teacher logits.
    
    Args:
        student_logits: (B, T, vocab_size) logits from student model
        teacher_logits: (B, T, vocab_size) logits from teacher model
        temperature: Temperature for softmax (higher = softer distribution)
        reduction: 'mean' or 'sum' or 'none'
    
    Returns:
        loss: Scalar or (B, T) tensor depending on reduction
    """
    # Apply temperature scaling
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL divergence: We use KL(teacher || student) which is more numerically stable
    # KL(teacher || student) = sum(teacher * log(teacher/student))
    # = sum(teacher * log(teacher)) - sum(teacher * log(student))
    # Using F.kl_div: input=log(student), target=teacher, log_target=False
    # This computes: sum(target * (log(target) - input))
    # = sum(teacher * (log(teacher) - log(student))) = KL(teacher || student)
    kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='none',
        log_target=False
    )  # (B, T, vocab_size)
    
    # Sum over vocab dimension
    kl_loss = kl_loss.sum(dim=-1)  # (B, T)
    
    # Scale by temperature^2 (standard in distillation literature)
    kl_loss = kl_loss * (temperature ** 2)
    
    # Sum over vocab dimension, then apply reduction
    kl_loss = kl_loss.sum(dim=-1)  # (B, T)
    
    if reduction == 'mean':
        return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()
    else:
        return kl_loss


def compute_combined_loss(
    student_logits,
    teacher_logits,
    targets,
    temperature=1.0,
    alpha=0.5,
    ignore_index=-1,
    reduction='mean'
):
    """
    Combine distillation loss with standard cross-entropy loss.
    
    Args:
        student_logits: (B, T, vocab_size) logits from student model
        teacher_logits: (B, T, vocab_size) logits from teacher model
        targets: (B, T) ground truth token ids
        temperature: Temperature for distillation
        alpha: Weight for distillation loss (1-alpha for CE loss)
        ignore_index: Tokens to ignore in CE loss
        reduction: 'mean' or 'sum' or 'none'
    
    Returns:
        total_loss: Combined loss
        distill_loss: Distillation loss component
        ce_loss: Cross-entropy loss component
    """
    # Distillation loss
    distill_loss = compute_distillation_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
        reduction=reduction
    )
    
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction=reduction
    )
    
    # Combine: alpha * distill + (1-alpha) * ce
    if reduction == 'none':
        # For 'none', we need to handle the shape mismatch
        # distill_loss is (B, T), ce_loss is (B*T,)
        ce_loss = ce_loss.view(student_logits.shape[:2])
        total_loss = alpha * distill_loss + (1 - alpha) * ce_loss
    else:
        total_loss = alpha * distill_loss + (1 - alpha) * ce_loss
    
    return total_loss, distill_loss, ce_loss


def compute_multi_token_loss(multi_token_logits, targets, ignore_index=-1, reduction='mean'):
    """Train multi-token heads (t+2, t+3, t+4 predictions)"""
    total_loss = 0.0
    count = 0
    
    for head_name, logits in multi_token_logits.items():
        offset = int(head_name.split('_')[1])  # "head_2" -> 2
        
        # Shift targets: head_2 predicts t+2, so target is y shifted by 1
        if targets.size(1) >= offset:
            shifted_targets = targets[:, offset-1:]
            shifted_logits = logits[:, :targets.size(1)-offset+1, :]
            
            if shifted_targets.numel() > 0:
                loss = F.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.size(-1)),
                    shifted_targets.reshape(-1),
                    ignore_index=ignore_index,
                    reduction=reduction
                )
                total_loss += loss
                count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0, device=targets.device)


def compute_draft_loss(student_model, x, teacher_logits, temperature=1.0):
    """Train draft head to predict multiple future tokens"""
    if student_model.draft_head is None:
        return torch.tensor(0.0, device=x.device)
    
    # Get hidden states from last transformer layer
    from nanochat.gpt import norm
    hidden = student_model.transformer.wte(x)
    hidden = norm(hidden)
    x0 = hidden
    
    for i, block in enumerate(student_model.transformer.h):
        hidden = student_model.resid_lambdas[i] * hidden + student_model.x0_lambdas[i] * x0
        ve = student_model.value_embeds[str(i)](x) if str(i) in student_model.value_embeds else None
        cos_sin = student_model.cos[:, :x.size(1)], student_model.sin[:, :x.size(1)]
        hidden = block(hidden, ve, cos_sin, student_model.window_sizes[i], None)
    
    hidden = norm(hidden)
    last_hidden = hidden[:, -1, :]  # (B, n_embd)
    
    # Draft head predicts next N tokens
    draft_logits = student_model.draft_head(last_hidden)  # (B, draft_n, vocab)
    
    # Match with teacher's future predictions
    B, T, V = teacher_logits.shape
    draft_n = draft_logits.shape[1]
    
    total_loss = 0.0
    for i in range(min(draft_n, T-1)):
        draft_pred = draft_logits[:, i, :]
        teacher_future = teacher_logits[:, i+1, :]
        
        student_log_probs = F.log_softmax(draft_pred / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_future / temperature, dim=-1)
        
        kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean', log_target=False)
        total_loss += kl
    
    return total_loss / min(draft_n, T-1) if T > 1 else torch.tensor(0.0, device=x.device)

