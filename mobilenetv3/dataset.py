import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from mobilenetv3.Classes import CLASSES  # 전체 클래스 목록

class MultiLabelDataset(Dataset):
    def __init__(self, image_dir, json_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = []

        # 클래스 이름 → 정수 인덱스 매핑
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        for filename, tag_list in json_data.items():
            label_vec = [0] * len(CLASSES)
            for tag in tag_list:
                if tag in self.class_to_idx:
                    label_vec[self.class_to_idx[tag]] = 1
            self.data.append((filename, torch.FloatTensor(label_vec)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
