import os
import shutil
import random
from PIL import Image
from tqdm import tqdm

from torchvision import transforms

# 원본 폴더와 대상 폴더 경로
SOURCE_DIR = "dataset_raw"
TARGET_DIR = "dataset"
IMG_SIZE = (224, 224)

# 비율 설정
split_ratio = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

# 전처리 파이프라인 정의
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
])

def prepare_dataset():
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    for split in split_ratio.keys():
        os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

    class_names = os.listdir(SOURCE_DIR)

    for class_name in class_names:
        class_path = os.path.join(SOURCE_DIR, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * split_ratio["train"])
        n_val = int(n * split_ratio["val"])
        n_test = n - n_train - n_val

        split_counts = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:],
        }

        for split, files in split_counts.items():
            split_dir = os.path.join(TARGET_DIR, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for file in tqdm(files, desc=f"{class_name} -> {split}"):
                src = os.path.join(class_path, file)
                dst = os.path.join(split_dir, file)
                try:
                    img = Image.open(src).convert("RGB")
                    img = transform(img)
                    img.save(dst)
                except Exception as e:
                    print(f"[ERROR] {file}: {e}")

if __name__ == "__main__":
    prepare_dataset()