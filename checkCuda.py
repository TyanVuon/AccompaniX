import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Print the local CUDA version
    cuda_version = torch.version.cuda
    print(f"Local CUDA version: {cuda_version}")

    # Print the PyTorch version
    pytorch_version = torch.__version__
    print(f"PyTorch version: {pytorch_version}")

    # Check if PyTorch is using CUDA
    if torch.version.cuda is not None:
        print("PyTorch is compiled with CUDA support.")
    else:
        print("PyTorch is not compiled with CUDA support.")
else:
    print("CUDA is not available on your system.")

if torch.cuda.is_available():
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
else:
    print("cuDNN is not available.")