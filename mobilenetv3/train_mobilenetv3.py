import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from mobilenetv3.dataset import MultiLabelDataset
from mobilenetv3.Classes import CLASSES
import onnx

# 설정
BATCH_SIZE = 32  # 한번에 학습할 이미지 수
EPOCHS = 10  # 전체 데이터셋 반복 학습할 횟수수
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cpu")  # CPU로 학습 진행

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 데이터셋
train_ds = MultiLabelDataset(
    image_dir="dataset/train",
    json_path="multitag_train.json",
    transform=transform
)

val_ds = MultiLabelDataset(
    image_dir="dataset/val",
    json_path="multitag_val.json",
    transform=transform
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
# 모델 정의
model = models.mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# 손실 함수 & 옵티마이저
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {running_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), "mobilenetv3/mobilenetv3_trained.pth")

# ONNX 저장
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
torch.onnx.export(model, dummy_input, "mobilenetv3/mobilenetv3_trained.onnx",
                  input_names=["input"], output_names=["output"],
                  export_params=True, opset_version=11)

print("✅ 모델 학습 및 ONNX 변환 완료")