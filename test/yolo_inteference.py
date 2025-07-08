from ultralytics import YOLO
import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU is available. Using GPU.")
else:
    device = 'cpu'
    print("GPU is not available. Using CPU.")

# model = YOLO('models/yolo5_last.pt')  # Load model
# result = model.predict('input_videos/input_video.mp4', save=True,conf=0.2, device=device)  # Inference image


# print(result)  # print results
# print("BOXES: ")  # print boxes
# for box in result[0].boxes:
#     print(box)