from huggingface_hub import snapshot_download

model_id = "IDEA-Research/grounding-dino-tiny"
snapshot_download(repo_id=model_id, repo_type="model")