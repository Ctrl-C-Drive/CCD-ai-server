import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from clip.koclip_model import encode_text

# 현재 디렉토리에서 import
from clip_api import vectorize_image_by_path, delete_image_vector

# 1. 환경 변수 로드
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path=env_path)

# 2. Pinecone 초기화 및 인덱스 연결
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("images")

# 3. 테스트용 UUID와 이미지 경로
uuid = "test-uuid-001"
image_path = "C:\\Github\\CCD\\CCD-ai-server\\uploads\\original\\6acdecb4-2873-45dc-b127-7b9db5f4d87b.png"

# 4. 벡터 저장 테스트
print("=== [1] 벡터 저장 ===")
result = vectorize_image_by_path(uuid, image_path)
print(result)

# 5. 저장된 벡터 가져오기
print("\n=== [2] fetch 결과 ===")
fetch_result = index.fetch(ids=[uuid])
print(fetch_result)

# 6. 인덱스 상태 출력
print("\n=== [3] describe_index_stats 결과 ===")
print(index.describe_index_stats())

# 7. 텍스트 임베딩 테스트
print("\n=== [4] 텍스트 임베딩 테스트 ===")
test_text = "레이싱 자동차"

try:
    embedding = encode_text(test_text)
    print(f"임베딩 벡터 길이: {embedding.shape}")
    print("nan 포함 여부:", torch.isnan(embedding).any().item())
    print("inf 포함 여부:", torch.isinf(embedding).any().item())
    print(f"첫 5개 값: {embedding[:5]}")
except Exception as e:
    print("❌ 텍스트 임베딩 실패:", str(e))
    import traceback
    traceback.print_exc()

# 8. Pinecone 텍스트 검색 테스트
print("\n=== [5] Pinecone 검색 테스트 ===")
try:
    search_result = index.query(
        vector=embedding.tolist(),
        top_k=5,
        include_metadata=True
    )
    print("검색 결과 개수:", len(search_result['matches']))
    for match in search_result['matches']:
        print(f"ID: {match.get('id')}, Score: {match.get('score')}")
except Exception as e:
    print("❌ Pinecone 검색 실패:", str(e))
    import traceback
    traceback.print_exc()
