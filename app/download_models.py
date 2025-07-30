
from huggingface_hub import hf_hub_download
import os

MODEL_REPO = "Gladiator1192/Backend-Model"
MODEL_FILENAMES = [
    "classification_model.pth",
    "card_detect.pt",
    "dob.pt",
    "gender.pt",
    "name.pt",
    "address.pt"
]

def download_all_models():
    os.makedirs("models", exist_ok=True)
    for model_name in MODEL_FILENAMES:
        print(f"Downloading {model_name} from {MODEL_REPO}...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=model_name,
            cache_dir="models",
            force_download=False
        )
        print(f"Saved to: {model_path}")

if __name__ == "__main__":
    download_all_models()
