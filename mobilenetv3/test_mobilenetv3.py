import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from mobilenetv3.Classes import CLASSES

# 설정
MODEL_PATH = "mobilenetv3_trained_newnew.onnx"
IMAGE_PATH = "C:/Users/parks/Desktop/jonjal.jpg"  # ✅ 테스트할 이미지 경로

THRESHOLD = 0.5  # 다중 태깅 확률 기준값

# 전처리 transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 이미지 로딩
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).numpy()  # shape: (1, 3, 224, 224)

# ONNX 모델 로드
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 추론
output = session.run([output_name], {input_name: input_tensor})[0]
probs = torch.sigmoid(torch.tensor(output)).squeeze().tolist()

# 다중 태깅 결과
predicted_tags = [CLASSES[i] for i, p in enumerate(probs) if p >= THRESHOLD]

# 출력
print(f"✅ 예측 태그: {predicted_tags}")
print(f"📷 테스트 이미지: {os.path.basename(IMAGE_PATH)}")