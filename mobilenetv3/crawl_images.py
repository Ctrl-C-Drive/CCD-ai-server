import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from mobilenetv3.Classes import CLASSES
import os

KEYWORDS = ["개"]
SAVE_DIR = "dataset_raw"
TARGET_PER_CLASS = 200
MAX_TRIES_PER_KEYWORD = 1  # 각 키워드당 최대 1회 크롤링

os.makedirs(SAVE_DIR, exist_ok=True)

def get_image_count(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

def crawl_images(cls_dir, keyword, max_num):
    print(f"[INFO] '{keyword}'로 최대 {max_num}장 수집 시도 중...")
    crawler = GoogleImageCrawler(storage={"root_dir": cls_dir})
    crawler.crawl(keyword=keyword, max_num=max_num)

for cls in KEYWORDS:
    cls_dir = os.path.join(SAVE_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)

    current_count = get_image_count(cls_dir)
    remaining = TARGET_PER_CLASS - current_count

    keywords = [
        cls + "사진"
    ]

    for keyword in keywords:

        if remaining <= 0:
            break

        crawl_images(cls_dir, keyword, remaining)
        current_count = get_image_count(cls_dir)
        remaining = TARGET_PER_CLASS - current_count

    # 보충 시도 (cls 키워드로 다시)
    if remaining > 0:
        print(f"[RETRY] {cls} - 부족한 {remaining}장을 '{cls}' 키워드로 한 번 더 수집 시도...")
        crawl_images(cls_dir, cls, remaining)

    # 마지막 보충 시도 (Bing)
    if remaining > 0:
        print(f"[BING-FALLBACK] {cls} - Bing으로 {remaining}장 수집 시도...")
        bing = BingImageCrawler(storage={"root_dir": cls_dir})
        bing.crawl(keyword=cls, max_num=remaining)

    final_count = get_image_count(cls_dir)
    print(f"[DONE] {cls} - 최종 {final_count}장 수집 완료\n{'='*50}")