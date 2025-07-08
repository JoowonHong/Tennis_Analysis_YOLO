import torch


print(f"CUDA 사용 가능: {torch.cuda.is_available()}")  # True 출력 확인[1][5]
print(f"GPU 이름: {torch.cuda.get_device_name(0)}")

print(torch.__version__)  # 1.12.0 이상 권장
print(torch.version.cuda)  # 11.3 이상 확인[21]