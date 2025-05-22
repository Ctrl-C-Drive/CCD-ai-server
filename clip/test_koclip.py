import os
import torch
from koclip_model import encode_text, encode_image

# 설정
image_dir = "./images"  # 이미지 폴더
query = "고양이 사진"                # 검색어

# 이미지 파일 로딩
image_paths = [
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.lower().endswith((".jpg", ".png", ".jpeg"))
]

# 텍스트 임베딩
text_vec = encode_text(query).unsqueeze(0)  # (1, 1024)

# 이미지 임베딩
image_vecs = []
valid_paths = []

for path in image_paths:
    try:
        img_vec = encode_image(path)
        image_vecs.append(img_vec)
        valid_paths.append(path)
    except Exception as e:
        print(f"{path} 실패: {e}")

image_vecs = torch.stack(image_vecs)  # (N, 1024)

# cosine similarity (Pure PyTorch)
text_vec = torch.nn.functional.normalize(text_vec, dim=-1)
image_vecs = torch.nn.functional.normalize(image_vecs, dim=-1)
similarities = torch.matmul(text_vec, image_vecs.T)[0]  # (N,)

# Top-K 결과
topk = 5
top_indices = similarities.argsort(descending=True)[:topk]

# 출력
print(f"\n검색어: \"{query}\"")
print("가장 유사한 이미지 TOP5:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. {valid_paths[idx]} → 유사도: {similarities[idx]:.4f}")