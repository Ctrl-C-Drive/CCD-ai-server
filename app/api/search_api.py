from fastapi import APIRouter
from pydantic import BaseModel
from clip.koclip_model import encode_text
import pinecone
import os
from dotenv import load_dotenv

# 0. 환경 변수 로딩
load_dotenv()

# 1. Pinecone 초기화
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

index_name = "image"
if index_name not in pinecone.list_indexes():
    raise ValueError(f"Pinecone 인덱스 '{index_name}'가 존재하지 않습니다.")
index = pinecone.Index(index_name)

# 2. FastAPI 라우터 등록
router = APIRouter()

# 3. 요청 바디 스키마
class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5  # 기본값 설정

@router.post("/search-text")
def search_similar_images(data: TextSearchRequest):
    query = data.query
    top_k = data.top_k

    # 4. KoCLIP 텍스트 임베딩
    vec = encode_text(query).tolist()

    # 5. Pinecone에서 유사 벡터 검색
    result = index.query(vector=vec, top_k=top_k, include_metadata=True)

    # 6. 결과 가공
    matches = []
    for match in result['matches']:
        matches.append({
            "score": match["score"],
            "uuid": match["id"],
            "filename": match["metadata"].get("filename"),
            "path": match["metadata"].get("path"),
        })

    return {
        "query": query,
        "results": matches
    }
