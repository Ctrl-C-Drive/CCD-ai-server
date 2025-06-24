import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

KEYWORDS = ["강아지", "고양이"]
SAVE_DIR = "dataset_raw"
TARGET_PER_CLASS = 200

os.makedirs(SAVE_DIR, exist_ok=True)

def get_image_count(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

def crawl_images(cls_dir, keyword, max_num):
    print(f"[INFO] '{keyword}'로 최대 {max_num}장 수집 시도 중...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    search_url = f"https://search.naver.com/search.naver?where=image&query={keyword}&sm=tab_opt"
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # 이미지 태그를 찾는 부분 수정
    image_tags = soup.find_all("img")
    image_urls = [img["src"] for img in image_tags if "src" in img.attrs and img["src"].startswith("http")]

    if not image_urls:
        print(f"[WARNING] '{keyword}'에 대한 이미지를 찾을 수 없습니다.")
        return

    for idx, image_url in enumerate(tqdm(image_urls[:max_num], desc=f"Downloading {keyword}")):
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            image_path = os.path.join(cls_dir, f"{keyword}_{idx + 1}.jpg")
            with open(image_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        except Exception as e:
            print(f"Failed to download {image_url}: {e}")

for cls in KEYWORDS:
    cls_dir = os.path.join(SAVE_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)

    current_count = get_image_count(cls_dir)
    remaining = TARGET_PER_CLASS - current_count

    keyword = f"{cls} 사진"

    if remaining > 0:
        crawl_images(cls_dir, keyword, remaining)

    final_count = get_image_count(cls_dir)
    print(f"[DONE] {cls} - 최종 {final_count}장 수집 완료\n{'='*50}")