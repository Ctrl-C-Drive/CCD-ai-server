import os
import shutil
from tqdm import tqdm
from PIL import Image

# 원본 폴더와 대상 폴더 경로
SOURCE_DIR = "dataset_raw"
TARGET_DIR = "dataset"

def preprocess_images():
    # 대상 폴더 생성
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    # 원본 폴더의 모든 클래스 폴더를 순회
    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue  # 폴더가 아닌 경우 건너뜀

        # 클래스별 폴더 생성
        target_class_dir = os.path.join(TARGET_DIR, class_name)
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        # 클래스 폴더 내의 모든 이미지 파일을 순회
        for image_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            image_path = os.path.join(class_path, image_name)
            if not os.path.isfile(image_path):
                continue  # 파일이 아닌 경우 건너뜀

            try:
                # 이미지 열기 및 전처리 (예: 크기 조정)
                with Image.open(image_path) as img:
                    img = img.convert("RGB")  # RGB로 변환
                    img = img.resize((224, 224))  # 크기 조정

                    # 대상 클래스 폴더로 이미지 저장
                    target_path = os.path.join(target_class_dir, image_name)
                    img.save(target_path)
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")

if __name__ == "__main__":
    preprocess_images()