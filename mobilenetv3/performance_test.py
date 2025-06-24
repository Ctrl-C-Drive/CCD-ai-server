import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import torch
import onnxruntime as ort
from torchvision import transforms
from mobilenetv3.Classes import CLASSES

# 설정
MODEL_PATH = "mobilenetv3_trained_newnewnew.onnx"
BASE_SAVE_DIR = "performance_test_images"
NUM_IMAGES = 150
THRESHOLD = 0.5  # 다중 태깅 확률 기준값

# 전처리 transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 1. Google에서 이미지 크롤링
def crawl_images(keyword, save_dir, max_num):
    print(f"[INFO] '{keyword}'로 최대 {max_num}장 수집 시도 중...")
    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=keyword, max_num=max_num)

# 2. MobileNetV3 모델 로드
print(f"[STEP 2] MobileNetV3 모델 로드 중...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"[STEP 2 완료] 모델 로드 완료.")

# 3. 성능 테스트 함수
def test_performance_for_keyword(keyword):
    # 키워드별 폴더 생성
    save_dir = os.path.join(BASE_SAVE_DIR, keyword)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        for file in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, file))

    # 이미지 크롤링
    # print(f"[STEP 1] Google에서 '{keyword}' 키워드로 이미지 크롤링 시작...")
    # crawl_images(keyword, save_dir, NUM_IMAGES)
    # print(f"[STEP 1 완료] 이미지 크롤링 완료. 저장 경로: {save_dir}")

    # 성능 테스트
    print(f"[STEP 3] '{keyword}' 키워드에 대한 성능 테스트 시작...")
    start_time = time.time()
    correct_predictions = 0
    total_predictions = 0

    for image_name in os.listdir(save_dir):
        image_path = os.path.join(save_dir, image_name)
        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).numpy()  # shape: (1, 3, 224, 224)

            # 추론
            output = session.run([output_name], {input_name: input_tensor})[0]
            probs = torch.sigmoid(torch.tensor(output)).squeeze().tolist()

            # 예측 태그
            predicted_tags = [CLASSES[i] for i, p in enumerate(probs) if p >= THRESHOLD]

            # 성능 평가 (예: 현재 키워드가 예측 태그에 포함되는지 확인)
            if keyword in predicted_tags:
                correct_predictions += 1
            total_predictions += 1

        except Exception as e:
            print(f"[ERROR] 이미지 처리 실패: {image_name}, 이유: {e}")

    end_time = time.time()

    # 결과 계산
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    elapsed_time = end_time - start_time

    print(f"[STEP 3 완료] '{keyword}' 키워드에 대한 성능 테스트 완료")
    print(f"총 테스트 이미지 수: {total_predictions}")
    print(f"'{keyword}' 태그 정확도: {accuracy:.2f}%")
    print(f"총 소요 시간: {elapsed_time:.2f}초")
    print("=" * 50)

    return keyword, accuracy, elapsed_time

# 4. CLASSES 순회하며 성능 테스트
results = []
for keyword in CLASSES:
    result = test_performance_for_keyword(keyword)
    results.append(result)

# 5. 최종 결과 출력
print("\n[최종 결과]")
for keyword, accuracy, elapsed_time in results:
    print(f"키워드: {keyword}, 정확도: {accuracy:.2f}%, 소요 시간: {elapsed_time:.2f}초")