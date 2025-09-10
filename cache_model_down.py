from huggingface_hub import snapshot_download

model_id = "IDEA-Research/grounding-dino-tiny"
cache_dir = "./local_model_cache"

local_model_path = snapshot_download(repo_id=model_id,
                                     repo_type="model",
                                     cache_dir=cache_dir)

print(f"Local model cached at: {local_model_path}")