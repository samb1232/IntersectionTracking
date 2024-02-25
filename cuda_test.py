import torch

# Проверка доступности CUDA
cuda_available = torch.cuda.is_available()

# Вывод информации о GPU, если CUDA доступен
if cuda_available:
    cuda_device = torch.cuda.get_device_name(0)
    print(f"CUDA is available. GPU: {cuda_device}")
else:
    print("CUDA is not available. Running on CPU.")
