import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from pinecone import Pinecone

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