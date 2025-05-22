# 패키지 및 모델 로드
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

repo = "Bingsu/clip-vit-large-patch14-ko"
model = CLIPModel.from_pretrained(repo)
processor = CLIPProcessor.from_pretrained(repo)
device = "cpu"

# 텍스트 임베딩 함수
def encode_text(text: str) -> torch.Tensor:
    """
    주어진 한국어 텍스트를 KoCLIP 임베딩 벡터로 변환
    """
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    with torch.inference_mode():
        outputs = model.get_text_features(**inputs)
    return outputs[0]

# 이미지 임베딩 함수
def encode_image(image_path: str) -> torch.Tensor:
    """
    주어진 이미지 파일 경로를 KoCLIP 임베딩 벡터로 변환
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    with torch.inference_mode():
        outputs = model.get_image_features(**inputs)
    return outputs[0]