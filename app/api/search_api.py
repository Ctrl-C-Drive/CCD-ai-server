from fastapi import APIRouter
from pydantic import BaseModel
from clip.koclip_model import encode_text
import os
import pinecone
from dotenv import load_dotenv

# 0. 환경 변수 로딩
load_dotenv()

# 1. Pinecone 초기화
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

index = pinecone.Index("image")

# 2. FastAPI 라우터 등록
router = APIRouter()

# 3. 요청 바디 스키마
class TextSearchRequest(BaseModel):
    query: str

@router.post("/search-text")
def search_similar_images(data: TextSearchRequest):
    query = data.query

    # 4. KoCLIP 텍스트 임베딩
    vec = encode_text(query).tolist()

    # 5. Pinecone에서 유사 벡터 검색
    result = index.query(vector=vec, include_metadata=True)

    # 6. 결과 가공
    matches = []
    for match in result['matches']:
        matches.append(match["id"])  # ID만 추가

    return {
        "query": query,
        "results": matches  # 결과를 ID 리스트로 반환
    }