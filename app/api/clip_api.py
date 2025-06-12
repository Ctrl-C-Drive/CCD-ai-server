import sys
import os
from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from clip.koclip_model import encode_image
from pinecone import Pinecone, ServerlessSpec  # ✅ 최신 SDK 방식

# 경로 설정 (clip 모듈 찾기용)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 0. 환경 변수 로딩
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=os.path.abspath(env_path))

# 1. Pinecone 객체 생성
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "images"

# 인덱스 없으면 생성
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# 인덱스 핸들 가져오기
index = pc.Index(index_name)

# 2. FastAPI 라우터 등록
router = APIRouter()

# 3. 이미지 벡터화 함수
def vectorize_image_by_path(user_id: str, image_uuid: str, image_path: str):
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
                "userid": user_id,
                "filename": os.path.basename(image_path),
                "path": image_path
            }
        }
    ])

    return {
        "message": "vector stored to Pinecone",
        "userid": user_id,
        "uuid": image_uuid,
        "path": image_path
    }

# 4. 삭제 함수
def delete_image_vector(user_id: str, image_uuid: str):
    try:
        # Pinecone에서 해당 벡터의 metadata 가져오기
        result = index.fetch(ids=[image_uuid])
        vector_data = result["vectors"].get(image_uuid)

        if not vector_data:
            return {"error": f"Vector with ID {image_uuid} not found in Pinecone"}

        # metadata에서 user_id 확인
        stored_user_id = vector_data["metadata"].get("userid")

        # user_id가 일치하면 삭제
        if stored_user_id == user_id:
            index.delete(ids=[image_uuid])
            return {"message": f"{image_uuid} deleted from Pinecone"}
    except Exception as e:
        return {"error": f"Failed to delete vector: {str(e)}"}

# 5. record 전체 삭제 함수
def delete_all_vectors():
    try:
        index.delete(delete_all=True)
        return {"message": "All vectors deleted from Pinecone"}
    except Exception as e:
        return {"error": f"Failed to delete all vectors: {str(e)}"}