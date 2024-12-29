import torch
try:
    import torch
    print(torch.version)
except ImportError as e:
    print("Error importing torch:", e)
