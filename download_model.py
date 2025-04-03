import os
import argparse
from huggingface_hub import snapshot_download

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID to download")
    parser.add_argument("--local_dir", type=str, default="models/", help="Local directory to save the downloaded model")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    model_id = args.model_id
    local_dir = os.path.join(args.local_dir, model_id)

    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(repo_id=model_id, local_dir=local_dir)
