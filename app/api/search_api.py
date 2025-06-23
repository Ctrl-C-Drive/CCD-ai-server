# import logging
# from typing import Dict
# from fastapi import APIRouter, Depends, HTTPException
# from pydantic import BaseModel
# from clip.koclip_model import encode_text
# import os
# import pinecone
# from dotenv import load_dotenv
# from core.dependencies import get_db, get_current_user

# # 0. 환경 변수 로딩
# load_dotenv()

# # 1. Pinecone 초기화
# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment=os.getenv("PINECONE_ENV")
# )

# index = pinecone.Index("image")

# # 2. FastAPI 라우터 등록
# router = APIRouter()

# # 3. 요청 바디 스키마
# class TextSearchRequest(BaseModel):
#     query: str


# @router.post("/search-text")
# async def search_similar_images(
#     data: TextSearchRequest,
#     user: Dict = Depends(get_current_user),
#     db=Depends(get_db)
# ):
#     query = data.query
#     user_id = user["user_id"]

#     # KoCLIP 텍스트 임베딩
#     vec = encode_text(query).tolist()

#     # Pinecone에서 유사 벡터 검색
#     result = index.query(
#         vector=vec, 
#         top_k=10,  # 상위 10개 결과
#         include_metadata=True
#     )
    
#     # 검색 결과에서 ID 추출
#     image_ids = [match["id"] for match in result['matches']]
    
#     # DB에서 상세 정보 조회
#     conn, cursor = db
#     try:
#         if not image_ids:
#             return {"query": query, "results": []}
        
#         # IN 절 파라미터 준비
#         placeholders = ",".join(["%s"] * len(image_ids))
#         query_sql = f"""
#             SELECT c.id, c.user_id, c.content, c.type, c.format, c.created_at,
#                    GROUP_CONCAT(t.tag_id) AS tag_ids,
#                    GROUP_CONCAT(t.name) AS tag_names,
#                    GROUP_CONCAT(t.source) AS tag_sources,
#                    im.width, im.height, im.file_size,
#                    im.file_path, im.thumbnail_path
#             FROM clipboard c
#             LEFT JOIN data_tag dt ON c.id = dt.data_id
#             LEFT JOIN tag t ON dt.tag_id = t.tag_id
#             LEFT JOIN image_meta im ON c.id = im.data_id
#             WHERE c.user_id = %s
#             AND c.id IN ({placeholders})
#             GROUP BY c.id
#             ORDER BY c.created_at DESC
#         """
#         params = [user_id] + image_ids
        
#         await cursor.execute(query_sql, params)
#         results = await cursor.fetchall()
        
#         # 결과 가공
#         data_list = []
#         for row in results:
#             tags = []
#             if row["tag_ids"]:
#                 tag_ids = row["tag_ids"].split(',')
#                 names = row["tag_names"].split(',')
#                 sources = row["tag_sources"].split(',')
#                 tags = [
#                     {"tag_id": tid, "name": n, "source": s}
#                     for tid, n, s in zip(tag_ids, names, sources)
#                 ]
            
#             image_meta = None
#             if row.get("file_path"):
#                 # 파일 경로에서 파일명 추출
#                 file_name = os.path.basename(row["file_path"])
#                 thumb_name = os.path.basename(row["thumbnail_path"]) if row.get("thumbnail_path") else None
                
#                 image_meta = {
#                     "width": row["width"],
#                     "height": row["height"],
#                     "file_size": row["file_size"],
#                     "file_path": f"/images/original/{file_name}",
#                     "thumbnail_path": f"/images/thumbnail/{thumb_name}" if thumb_name else None
#                 }
                
#             data_list.append({
#                 "id": row["id"],
#                 "user_id": row["user_id"],
#                 "content": row["content"],
#                 "data_type": row["type"],
#                 "format": row["format"],
#                 "created_at": row["created_at"],
#                 "tags": tags,
#                 "image_meta": image_meta
#             })
            
#         return {
#             "query": query,
#             "results": data_list
#         }
        
#     except Exception as e:
#         logging.error(f"Text search failed: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error")