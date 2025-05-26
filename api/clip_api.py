from fastapi import APIRouter
from pydantic import BaseModel
from clip.koclip_model import encode_image
import pinecone
import os
from dotenv import load_dotenv

# 0. 환경 변수 로딩
load_dotenv()

# 1. Pinecone 초기화
pinecone.init(
    api_key="pcsk_4DX7e9_KXt66tnRNAEZ1XEUxNwDgyFiqeyDgoEFV7e8HvWzY6k2hjXF3p8jTL2S2yNowBS",    
    environment=os.getenv("PINECONE_ENV")
)

index_name = "image"
if index_name not in pinecone.list_indexes():
    raise ValueError(f"Pinecone 인덱스 '{index_name}'가 존재하지 않습니다.")
index = pinecone.Index(index_name)

# 2. FastAPI 라우터 등록
router = APIRouter()

# 3. 요청 바디 스키마 정의
class ImageEmbeddingRequest(BaseModel):
    uuid: str           # 이미지 UUID
    path: str           # 이미지 실제 경로

@router.post("/vectorize-image")
def vectorize_image_by_path(data: ImageEmbeddingRequest):
    image_path = data.path
    image_uuid = data.uuid

    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

    # 4. KoCLIP 이미지 벡터 임베딩
    vec = encode_image(image_path)
    vec = vec.tolist()

    # 5. Pinecone에 업서트
    index.upsert([
        {
            "id": image_uuid,
            "values": vec,
            "metadata": {
                "filename": os.path.basename(image_path),
                "path": image_path
            }
        }
    ])

    return {
        "message": "vector stored to Pinecone",
        "uuid": image_uuid,
        "path": image_path
    }