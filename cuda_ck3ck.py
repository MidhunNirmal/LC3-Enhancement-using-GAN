import torch

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    torch.cuda.empty_cache()

else:
    print("CUDA is not available. Using CPU.")