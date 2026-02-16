"""
Push Finetuned Models to HuggingFace Hub
==========================================
Pushes both LLaMA and Gemma LoRA adapters to HuggingFace.
"""

import argparse
import os
from huggingface_hub import HfApi, login
from huggingface_hub import CommitOperationDelete as DeleteFileOperation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def push_model(adapter_path: str, repo_id: str, model_name: str):
    """Push a LoRA adapter to HuggingFace Hub."""
    api = HfApi()

    print(f"\nPushing {model_name} to {repo_id}...")

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
    except Exception as e:
        print(f"  Note: {e}")

    # Delete existing repo files first to avoid LFS pointer conflicts
    try:
        files = api.list_repo_files(repo_id)
        if files:
            delete_ops = [DeleteFileOperation(path_in_repo=f) for f in files if f != ".gitattributes"]
            if delete_ops:
                api.create_commit(
                    repo_id=repo_id,
                    operations=delete_ops,
                    commit_message="Clear old files before re-upload",
                )
                print(f"  Cleared {len(delete_ops)} old files from repo")
    except Exception as e:
        print(f"  Note (cleanup): {e}")

    # Upload the adapter directory
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        commit_message=f"Upload {model_name} LoRA adapter finetuned on smart home commands",
    )

    print(f"  ✓ Successfully pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push finetuned models to HuggingFace")
    parser.add_argument("--hf_username", type=str, required=True, help="HuggingFace username")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    # Login
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("No token provided. Make sure you're already logged in via `huggingface-cli login`.")

    # Push LLaMA
    llama_path = os.path.join(SCRIPT_DIR, "checkpoints_10k", "llama", "final")
    if os.path.exists(llama_path):
        push_model(
            adapter_path=llama_path,
            repo_id=f"{args.hf_username}/meta-llama-3-8b-smart-home-lora-10k",
            model_name="Meta-Llama-3-8B-Instruct",
        )
    else:
        print(f"LLaMA adapter not found at {llama_path}, skipping.")

    # Push Gemma
    gemma_path = os.path.join(SCRIPT_DIR, "checkpoints_10k", "gemma", "final")
    if os.path.exists(gemma_path):
        push_model(
            adapter_path=gemma_path,
            repo_id=f"{args.hf_username}/gemma-2-2b-smart-home-lora-10k",
            model_name="Gemma 2-2B-IT",
        )
    else:
        print(f"Gemma adapter not found at {gemma_path}, skipping.")

    # Also push results
    results_path = os.path.join(SCRIPT_DIR, "results")
    if os.path.exists(results_path):
        for model_name in ["llama", "gemma"]:
            repo_id = f"{args.hf_username}/{model_name.replace('llama', 'meta-llama-3-8b')}-smart-home-lora-10k"
            if model_name == "gemma":
                repo_id = f"{args.hf_username}/gemma-2-2b-smart-home-lora-10k"
            try:
                api = HfApi()
                results_file = os.path.join(results_path, "results.txt")
                if os.path.exists(results_file):
                    api.upload_file(
                        path_or_fileobj=results_file,
                        path_in_repo="results.txt",
                        repo_id=repo_id,
                    )
                    print(f"  ✓ Uploaded results.txt to {repo_id}")
            except Exception as e:
                print(f"  Could not upload results: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
