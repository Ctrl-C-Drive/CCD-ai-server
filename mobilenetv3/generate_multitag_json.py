import os
import json
from collections import defaultdict

def generate_json(root_dir, output_filename):
    tag_map = defaultdict(list)

    for tag_name in os.listdir(root_dir):
        tag_folder = os.path.join(root_dir, tag_name)
        if not os.path.isdir(tag_folder):
            continue
        for filename in os.listdir(tag_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                # 경로 포함해서 저장!
                relative_path = os.path.join(tag_name, filename).replace("\\", "/")
                tag_map[relative_path].append(tag_name)

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(tag_map, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_json("dataset/train", "multitag_train.json")
    generate_json("dataset/val", "multitag_val.json")
    generate_json("dataset/test", "multitag_test.json")
    print("✅ 저장 완료")