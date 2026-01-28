import gradio as gr
import torch
import pickle
from huggingface_hub import hf_hub_download

print("Downloading model files")
model_path = hf_hub_download(
    repo_id="TMVishnu/nanochat-distill-d12-int8",
    filename="model.pt"
)

tokenizer_pkl_path = hf_hub_download(
    repo_id="TMVishnu/nanochat-distill-d12-int8",
    filename="tokenizer.pkl"
)

token_bytes_path = hf_hub_download(
    repo_id="TMVishnu/nanochat-distill-d12-int8",
    filename="token_bytes.pt"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}")

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model_state = checkpoint["model"]
config = checkpoint["config"]

with open(tokenizer_pkl_path, "rb") as f:
    tokenizer = pickle.load(f)

token_bytes = torch.load(token_bytes_path, map_location=device, weights_only=False)

print(f"Model config: {config}")
print("Model loaded successfully")

def generate_text(prompt, max_tokens=50, temperature=0.8):
    try:
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        generated = tokens[0].tolist()
        
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = model_state(tokens)
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
                tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
                
                if next_token == tokenizer.eos_token_id:
                    break
        
        output = tokenizer.decode(generated)
        return output
        
    except Exception as e:
        return f"Error during generation: {str(e)}"

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt",
            lines=3
        ),
        gr.Slider(
            minimum=10,
            maximum=200,
            value=50,
            step=1,
            label="Max Tokens"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.8,
            step=0.1,
            label="Temperature"
        )
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="NanoChat Distilled Model INT8",
    description="375M parameter student model with MQA and INT8 quantization. Warning: Output quality is poor due to undertrained teacher model.",
    examples=[
        ["Explain what machine learning is", 100, 0.7],
        ["Write a short story about a robot", 150, 1.0],
        ["What is the capital of France", 50, 0.5],
        ["List three programming languages", 75, 0.8]
    ]
)
demo.launch()
