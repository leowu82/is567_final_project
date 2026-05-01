# Quickstart

Build the CUDA-enabled Docker image and run an interactive shell. Mount your local project and your `kaggle.json` into the container to allow dataset downloads.

Example (Windows PowerShell):

```powershell
docker build -t fraud-detect:latest .
docker run --gpus all -it \
  -v "C:\Users\User\Documents\kaggle.json:/root/.kaggle/kaggle.json:ro" \
  -v "${PWD}:/workspace" -w /workspace fraud-detect:latest
```

Inside the container:

```bash
conda activate fraud_env
python scripts/train_baseline.py
python scripts/run_inference.py --text "Please reset your password at http://example.com"
```
