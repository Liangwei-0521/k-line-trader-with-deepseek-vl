from huggingface_hub import snapshot_download

snapshot_download(repo_id="deepseek-ai/deepseek-small",
                  local_dir="/root/autodl-tmp",)