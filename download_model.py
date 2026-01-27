from huggingface_hub import snapshot_download
import os

cache_dir = os.path.expanduser("~/.cache/nanochat/base_checkpoints/d32")
print(f"Downloading to: {cache_dir}")

snapshot_download(
    repo_id="karpathy/nanochat-d32",
    local_dir=cache_dir,
    repo_type="model"
)

print("✅ Download complete!")
