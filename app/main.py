import asyncio
import io
from PIL import Image
import aiofiles
from fastapi import FastAPI, Form, HTTPException, Depends, status, Request, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Literal
from datetime import datetime, timedelta, timezone
import aiomysql
import uuid
import bcrypt
import jwt
import logging
from fastapi_limiter.depends import RateLimiter
from fastapi_limiter import FastAPILimiter
from redis.asyncio import Redis
import os

from clip.koclip_model import encode_text
from pinecone import Pinecone
from dotenv import load_dotenv

# 0. 환경 변수 로딩
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path=env_path)

# 1. Pinecone 초기화
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "images"
index = pc.Index(index_name)

# pinecone router 등록
from api.clip_api import router as clip_router

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        self.db_pool = None
        self.redis = None
        self.limiter = None

app_state = AppState()

# 환경 변수 설정
class Settings(BaseSettings):
    db_host: str = "mysql-db"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str
    db_name: str = "clipboard_db"
    redis_url: str = "redis://redis:6379"
    secret_key: str
    algorithm: Literal['HS256', 'HS384', 'HS512'] = 'HS256'
    access_token_expire_minutes: int = 1440
    refresh_token_expire_days: int = 7

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# 데이터 모델 
class UserSignupRequest(BaseModel):
    user_id: str = Field(..., min_length=4, max_length=50)
    password: str = Field(..., min_length=8)

class UserLoginRequest(UserSignupRequest):
    pass

class ItemCreate(BaseModel):
    id: str
    content: Optional[str] = None
    type: Literal["txt", "img"]
    format: str
    created_at: int

class ItemResponse(ItemCreate):
    id: str
    created_at: int

class RefreshTokenRequest(BaseModel):
    refresh_token: str


class MaxCountUpdateRequest(BaseModel):
    max_count_cloud: int = Field(..., ge=1, le=1000)

class TagCreateAndLink(BaseModel):
    data_id: str
    name: str
    source: str  # 'auto' or 'user'

class TagResponse(BaseModel):
    tag_id: str
    name: str
    source: str

class ClipboardDataResponse(BaseModel):
    id: str
    content: str
    type: str
    format: str
    created_at: int
    tags: List[TagResponse]
    image_meta: Optional[Dict] = None
class TextSearchRequest(BaseModel):
    query: str



# 데이터베이스 초기화 함수
async def initialize_database():
    DROP_TABLES = """
    DROP TABLE IF EXISTS data_tag;
    DROP TABLE IF EXISTS tag;
    DROP TABLE IF EXISTS image_meta;
    DROP TABLE IF EXISTS clipboard;
    DROP TABLE IF EXISTS refresh_token;
    DROP TABLE IF EXISTS user;
    """

    CREATE_TABLES = """
        CREATE TABLE IF NOT EXISTS user (
            user_id VARCHAR(255) PRIMARY KEY,
            password VARCHAR(255) NOT NULL,
            created_at INTEGER NOT NULL,
            max_count_cloud INTEGER NOT NULL DEFAULT 20
        );
        
        CREATE TABLE IF NOT EXISTS clipboard (
            id VARCHAR(36) PRIMARY KEY,  -- UUID는 항상 36자
            user_id VARCHAR(255) NOT NULL, 
            type ENUM('img', 'txt') NOT NULL,
            format VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES user(user_id) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS image_meta (
            data_id VARCHAR(36) PRIMARY KEY,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            file_path TEXT NOT NULL,
            thumbnail_path TEXT,
            FOREIGN KEY (data_id) REFERENCES clipboard(id) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS tag (
            tag_id VARCHAR(36) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            source ENUM('auto', 'user') NOT NULL,
            UNIQUE (name, source)
        );
        
        CREATE TABLE IF NOT EXISTS data_tag (
            data_id VARCHAR(36) NOT NULL,
            tag_id VARCHAR(36) NOT NULL,
            PRIMARY KEY (data_id, tag_id),
            FOREIGN KEY (data_id) REFERENCES clipboard(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tag(tag_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS refresh_token (
            jti VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            expires_at INTEGER NOT NULL,
            revoked BOOLEAN NOT NULL DEFAULT FALSE,
            FOREIGN KEY (user_id) REFERENCES user(user_id) ON DELETE CASCADE
        );
    """

    async with app_state.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                for statement in DROP_TABLES.strip().split(';'):
                    stmt = statement.strip()
                    if stmt:
                        await cursor.execute(stmt + ';')

                for statement in CREATE_TABLES.strip().split(';'):
                    stmt = statement.strip()
                    if stmt:
                        await cursor.execute(stmt + ';')

                await conn.commit()
            except Exception as e:
                await conn.rollback()
                logger.error(f"Database initialization failed: {str(e)}")
                if "already exists" not in str(e):
                    raise

# 애플리케이션 수명 주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # MySQL 연결 풀 생성
        app_state.db_pool = await aiomysql.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            db=settings.db_name,
            minsize=5,
            maxsize=20,
            auth_plugin="mysql_native_password",
            charset="utf8mb4",
            autocommit=True, 
        )

        # Redis 연결
        app_state.redis = Redis.from_url(settings.redis_url)
        await FastAPILimiter.init(app_state.redis)

        # 데이터베이스 초기화
        await initialize_database()

        yield

    except Exception as e:
        logger.critical(f"Application startup failed: {str(e)}")
        raise
    finally:
        # 리소스 정리
        if app_state.db_pool:
            app_state.db_pool.close()
            await app_state.db_pool.wait_closed()
        if app_state.redis:
            await app_state.redis.close()

app = FastAPI(lifespan=lifespan)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Create directories
original_dir = os.path.join(UPLOAD_DIR, "original")
thumbnail_dir = os.path.join(UPLOAD_DIR, "thumbnail")
os.makedirs(original_dir, exist_ok=True)
os.makedirs(thumbnail_dir, exist_ok=True)
app.mount("/images/original", StaticFiles(directory=original_dir), name="original-images")
app.mount("/images/thumbnail", StaticFiles(directory=thumbnail_dir), name="thumbnail-images")
app.include_router(clip_router)



# 의존성 주입
async def get_db():
    async with app_state.db_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            yield conn, cursor

security = HTTPBearer(auto_error=False)


# JWT 유틸리티 
def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode(
        {"sub": user_id, "exp": int(expire.timestamp()), "type": "access"},
        settings.secret_key,
        algorithm=settings.algorithm
    )

def create_refresh_token(user_id: str, jti: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    return jwt.encode(
        {"sub": user_id, "exp": int(expire.timestamp()), "type": "refresh", "jti": jti},
        settings.secret_key,
        algorithm=settings.algorithm
    )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    if not credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing credentials")
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        return {"user_id": payload["sub"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

# 엔드포인트 
@app.post("/signup")
async def signup(user: UserSignupRequest, db=Depends(get_db)):
    conn, cursor = db
    try:
        await cursor.execute("SELECT user_id FROM user WHERE user_id = %s", (user.user_id,))
        if await cursor.fetchone():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "User ID exists")

        hashed_pw = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
        await cursor.execute(
            "INSERT INTO user (user_id, password, created_at) VALUES (%s, %s, %s)",
            (user.user_id, hashed_pw, int(datetime.now(timezone.utc).timestamp()))
        )
        await conn.commit()
        return {"message": "User created"}

    except Exception as e:
        await conn.rollback()
        logger.error("Signup failed", exc_info=e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/login", dependencies=[Depends(RateLimiter(times=5, minutes=1))])
async def login(user: UserLoginRequest, db=Depends(get_db)):
    conn, cursor = db
    try:
        await cursor.execute(
            "SELECT password FROM user WHERE user_id = %s",
            (user.user_id,)
        )
        result = await cursor.fetchone()

        if not result or not bcrypt.checkpw(user.password.encode(), result["password"].encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        # 토큰 생성
        access_token = create_access_token(user.user_id)
        refresh_jti = str(uuid.uuid4())
        refresh_token = create_refresh_token(user.user_id, refresh_jti)

        # 리프레시 토큰 저장
        await cursor.execute(
            "INSERT INTO refresh_token (jti, user_id, expires_at, revoked) VALUES (%s, %s, %s, %s)",
            (refresh_jti, user.user_id, int((datetime.now(timezone.utc) + 
              timedelta(days=settings.refresh_token_expire_days)).timestamp()), False)
        )
        await conn.commit()

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    except HTTPException as e:
        logger.warning(f"Login failed with HTTP error: {e.detail}")
        raise e  # 꼭 다시 던져야 FastAPI가 제대로 처리

    except Exception as e:
        logger.error("Unexpected error during login", exc_info=e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@app.post("/refresh", dependencies=[Depends(RateLimiter(times=5, minutes=1))])
async def refresh_token(request: RefreshTokenRequest, db=Depends(get_db)):
    conn, cursor = db
    try:
        payload = jwt.decode(
            request.refresh_token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        if payload.get("type") != "refresh":
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token type")
        
        user_id = payload.get("sub")
        jti = payload.get("jti")
        current_time = int(datetime.now(timezone.utc).timestamp())

        # 리프레시 토큰 유효성 검사
        await cursor.execute(
            "SELECT jti FROM refresh_token WHERE jti = %s AND user_id = %s AND revoked = FALSE AND expires_at > %s",
            (jti, user_id, current_time)
        )
        token_record = await cursor.fetchone()
        if not token_record:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")

        # 기존 토큰 폐기
        await cursor.execute(
            "UPDATE refresh_token SET revoked = TRUE WHERE jti = %s",
            (jti,)
        )

        # 새 토큰 생성
        new_access_token = create_access_token(user_id)
        new_refresh_jti = str(uuid.uuid4())
        new_refresh_token = create_refresh_token(user_id, new_refresh_jti)

        # 새 리프레시 토큰 저장
        await cursor.execute(
            "INSERT INTO refresh_token (jti, user_id, expires_at, revoked) VALUES (%s, %s, %s, %s)",
            (new_refresh_jti, user_id, int((datetime.now(timezone.utc) + 
              timedelta(days=settings.refresh_token_expire_days)).timestamp()), False)
        )
        await conn.commit()

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    except Exception as e:
        await conn.rollback()
        logger.error("Refresh failed", exc_info=e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)
    
async def delete_with_update(user_id: str, max_count_cloud: int, conn, cursor):
    # cloud 항목 수 조회
    await cursor.execute(
        "SELECT id FROM clipboard WHERE user_id = %s ORDER BY created_at ASC",
        (user_id,)
    )
    items = await cursor.fetchall()
    if len(items) > max_count_cloud:
        # 초과한 항목 개수만큼 삭제
        to_delete = items[:len(items) - max_count_cloud]
        for item in to_delete:
            await cursor.execute("DELETE FROM clipboard WHERE id = %s", (item["id"],))

# 클라우드 최대 저장 개수 변경
@app.put("/user/max_count_cloud")
async def update_max_count_cloud(
    req: MaxCountUpdateRequest,
    current_user: Dict = Depends(get_current_user),
    db=Depends(get_db)
):
    conn, cursor = db
    user_id = current_user["user_id"]
    try:
        # 값 업데이트
        await cursor.execute(
            "UPDATE user SET max_count_cloud = %s WHERE user_id = %s",
            (req.max_count_cloud, user_id)
        )

        # 현재 사용자의 cloud 항목 수 조회
        await cursor.execute(
            "SELECT COUNT(*) AS cnt FROM clipboard WHERE user_id = %s",
            (user_id,)
        )
        result = await cursor.fetchone()
        current_count = result["cnt"]

        # 초과 시 삭제
        if current_count > req.max_count_cloud:
            await delete_with_update(user_id, req.max_count_cloud, conn, cursor)

        await conn.commit()
        return {"message": "max_count_cloud updated", "current_count": current_count}

    except Exception as e:
        await conn.rollback()
        logger.error("Failed to update max_count_cloud", exc_info=e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)

# 데이터가 post될 때마다 검사하여 최대 수를 안 넘기고 추가하게 함함
async def delete_oldest_item_if_full(user_id: str, max_count: int, conn, cursor):
    await cursor.execute(
        "SELECT COUNT(*) AS cnt FROM clipboard WHERE user_id = %s",
        (user_id,)
    )
    row = await cursor.fetchone()
    total_item_count = row['cnt']  # 또는 row[0] 

    if total_item_count >= max_count:
        # 삭제는 서브쿼리로 안전하게
        await cursor.execute(
            """
            DELETE FROM clipboard 
            WHERE id = (
                SELECT id FROM clipboard 
                WHERE user_id = %s 
                ORDER BY created_at ASC 
                LIMIT 1
            )
            """,
            (user_id,)
        )
        await conn.commit()
        logger.info(f"User {user_id}: Reached item limit. Deleted the oldest item.")

 
@app.post("/items", response_model=ItemResponse)
async def create_item(item: ItemCreate, user: Dict = Depends(get_current_user), db=Depends(get_db)):
    conn, cursor = db
    user_id = user["user_id"]
    try:
        await cursor.execute(
            "SELECT max_count_cloud FROM user WHERE user_id = %s",
            (user_id,)
        )
        user_settings = await cursor.fetchone()
        max_count_cloud = user_settings['max_count_cloud']

        # 꽉 찼다면 가장 오래된 아이템 삭제
        await delete_oldest_item_if_full(user_id, max_count_cloud, conn, cursor)
        await cursor.execute(
            """
            INSERT INTO clipboard (id, user_id, content, type, format, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                item.id,
                user_id,
                item.content,
                item.type,
                item.format,
                item.created_at,
            )
        )
        await conn.commit()

        return item.model_dump()

    except Exception as e:
        await conn.rollback()
        logger.error("Item creation failed", exc_info=e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@app.post("/tags")
async def create_tag_and_link(dt: TagCreateAndLink, db=Depends(get_db)):
    conn, cursor = db
    try:
        # 1. data_id 유효성 확인
        await cursor.execute("SELECT id FROM clipboard WHERE id = %s", (dt.data_id,))
        if not await cursor.fetchone():
            raise HTTPException(404, "Data not found")

        # 2. 동일한 (name, source) 태그가 있는지 확인
        await cursor.execute(
            "SELECT tag_id FROM tag WHERE name = %s AND source = %s",
            (dt.name, dt.source)
        )
        tag = await cursor.fetchone()

        if tag:
            tag_id = tag["tag_id"]
        else:
            # 3. 새 태그 생성
            tag_id = str(uuid.uuid4())
            await cursor.execute(
                "INSERT INTO tag (tag_id, name, source) VALUES (%s, %s, %s)",
                (tag_id, dt.name, dt.source)
            )

        # 4. data_tag 연결 시도
        try:
            await cursor.execute(
                "INSERT INTO data_tag (data_id, tag_id) VALUES (%s, %s)",
                (dt.data_id, tag_id)
            )
        except aiomysql.IntegrityError:
            # 이미 연결된 경우는 무시 가능
            pass

        await conn.commit()
        return {
            "tag_id": tag_id,
            "name": dt.name,
            "source": dt.source,
            "message": "Tag linked successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        await conn.rollback()
        logger.error(f"Tag link failed: {str(e)}")
        raise HTTPException(500, "Internal server error")
 
# @app.post("/tags", response_model=TagResponse)
# async def create_or_get_tag(tag: TagCreate, db=Depends(get_db)):
#     conn, cursor = db
#     try:
#         # 기존 태그 확인
#         await cursor.execute(
#             "SELECT tag_id FROM tag WHERE name = %s AND source = %s",
#             (tag.name, tag.source)
#         )
#         existing = await cursor.fetchone()
        
#         if existing:
#             return {
#                 "tag_id": existing["tag_id"],
#                 "name": tag.name,
#                 "source": tag.source
#             }

#         # 새 태그 생성 (tag_id가 전달되지 않았다면 새로 생성)
#         new_tag_id = getattr(tag, "tag_id", None) or str(uuid4())

#         await cursor.execute(
#             "INSERT INTO tag (tag_id, name, source) VALUES (%s, %s, %s)",
#             (new_tag_id, tag.name, tag.source)
#         )
#         await conn.commit()

#         return {
#             "tag_id": new_tag_id,
#             "name": tag.name,
#             "source": tag.source
#         }

#     except aiomysql.IntegrityError as e:
#         await conn.rollback()
#         if "Duplicate entry" in str(e):
#             raise HTTPException(400, "Tag ID already exists")
#         raise
#     except Exception as e:
#         await conn.rollback()
#         logger.error(f"Tag creation failed: {str(e)}")
#         raise HTTPException(500, "Internal server error")

# # 2. 데이터-태그 연결 엔드포인트
# @app.post("/data-tags")
# async def create_data_tag(dt: DataTagCreate, db=Depends(get_db)):
#     conn, cursor = db
#     try:
#         # 데이터 및 태그 존재 여부 확인
#         await cursor.execute("SELECT id FROM clipboard WHERE id = %s", (dt.data_id,))
#         if not await cursor.fetchone():
#             raise HTTPException(404, "Data not found")
            
#         await cursor.execute("SELECT tag_id FROM tag WHERE tag_id = %s", (dt.tag_id,))
#         if not await cursor.fetchone():
#             raise HTTPException(404, "Tag not found")
            
#         # 연결 생성
#         await cursor.execute(
#             "INSERT INTO data_tag (data_id, tag_id) VALUES (%s, %s)",
#             (dt.data_id, dt.tag_id)
#         )
#         await conn.commit()
#         return {"message": "Data-Tag association created"}
        
#     except aiomysql.IntegrityError:
#         await conn.rollback()
#         raise HTTPException(400, "Association already exists")
#     except Exception as e:
#         await conn.rollback()
#         logger.error(f"Data-Tag creation failed: {str(e)}")
#         raise HTTPException(500, "Internal server error")
    
from api.clip_api import vectorize_image_by_path, delete_image_vector

@app.post("/items/image")
async def upload_image(
    file: UploadFile = File(...),
    id: str = Form(...),
    format: str = Form(...),
    created_at: int = Form(...),
    user: Dict = Depends(get_current_user),
    db=Depends(get_db)
):
    conn, cursor = db
    user_id = user["user_id"]

    try:
        # Validate UUID
        uuid.UUID(id)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid ID format")

    try:
        # Check existing item
        await cursor.execute("SELECT id FROM clipboard WHERE id = %s", (id,))
        if await cursor.fetchone():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "ID already exists")

        # Get user settings
        await cursor.execute(
            "SELECT max_count_cloud FROM user WHERE user_id = %s",
            (user_id,)
        )
        max_count_cloud = (await cursor.fetchone())["max_count_cloud"]

        # Check and delete oldest items if needed
        await delete_oldest_item_if_full(user_id, max_count_cloud, conn, cursor)

        # Process image
        contents = await file.read()
        file_size = len(contents)
        loop = asyncio.get_running_loop()

        # Validate and process image
        try:
            image = await loop.run_in_executor(
                None, 
                lambda: Image.open(io.BytesIO(contents))
            )
            width, height = image.size
        except Exception:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid image file")

        
        # Generate filenames
        ext = file.filename.split('.')[-1] if '.' in file.filename else 'bin'
        original_filename = f"{id}.{ext}"
        thumbnail_filename = f"{id}_thumb.{ext}"
        original_path = os.path.join(original_dir, original_filename)
        thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
        original_url = f"/images/original/{original_filename}"
        thumbnail_url = f"/images/thumbnail/{thumbnail_filename}"

        # Save original file
        async with aiofiles.open(original_path, 'wb') as f:
            await f.write(contents)

        # Generate and save thumbnail
        def generate_thumbnail():
            thumb = image.copy()
            thumb.thumbnail((256, 256))
            thumb.save(thumbnail_path)

        await loop.run_in_executor(None, generate_thumbnail)

        # Database operations
        await cursor.execute(
            """
            INSERT INTO clipboard (id, user_id, content, type, format, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (id, user_id, original_url, 'img', format, created_at)
        )

        await cursor.execute(
            """
            INSERT INTO image_meta (data_id, width, height, file_size, file_path, thumbnail_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (id, width, height, file_size, original_url, thumbnail_url)
        )

        await conn.commit()
        
        try:
            # 동기 함수를 비동기 컨텍스트에서 실행
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                lambda: vectorize_image_by_path(id, original_path)
            )
        except Exception as e:
            logger.error(f"Vectorization failed: {str(e)}")

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"id": id, "message": "Image uploaded successfully"}
        )

    except HTTPException:
        raise
    except Exception as e:
        await conn.rollback()
        # Cleanup files if they were created
        for path in [original_path, thumbnail_path]:
            if path and os.path.exists(path):
                os.remove(path)
        logger.error(f"Image upload failed: {str(e)}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)

 # 클립보드 데이터 삭제
@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(
    item_id: str, 
    user: Dict = Depends(get_current_user), 
    db=Depends(get_db)
):
    conn, cursor = db
    try:
        # ON DELETE CASCADE로 인해 연관 데이터 자동 삭제
        await cursor.execute(
            "DELETE FROM clipboard WHERE id = %s",
            (item_id,)
        )
        
        if cursor.rowcount == 0:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail=f"Item {item_id} not found"
            )
            
        await conn.commit()
        
    except HTTPException:
        await conn.rollback()
        raise
    try:
        # 벡터 삭제
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, 
            lambda: delete_image_vector(item_id)
        )
    except Exception as e:
        logger.error(f"Vector deletion failed: {str(e)}")
        
    except Exception as e:
        await conn.rollback()
        logger.error(f"Item deletion failed: {str(e)}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Item deletion failed"
        )


# 사용자 데이터 조회
@app.get("/clipboard-data", response_model=List[ClipboardDataResponse])
async def get_user_clipboard_data(
    user: Dict = Depends(get_current_user), 
    db=Depends(get_db)
):

    conn, cursor = db
    await cursor.execute("SELECT DATABASE();")
    row = await cursor.fetchone()
    logger.info("FASTAPI uses DB: %s", row["DATABASE()"])   
    
    try:
        # 클립보드 데이터 + 태그 조회
        await cursor.execute("""
            SELECT c.*, 
                   GROUP_CONCAT(t.tag_id) AS tag_ids,
                   GROUP_CONCAT(t.name) AS tag_names,
                   GROUP_CONCAT(t.source) AS tag_sources,
                   im.width, im.height, im.file_size,
                   im.file_path, im.thumbnail_path
            FROM clipboard c
            LEFT JOIN data_tag dt ON c.id = dt.data_id
            LEFT JOIN tag t ON dt.tag_id = t.tag_id
            LEFT JOIN image_meta im ON c.id = im.data_id
            WHERE c.user_id = %s
            GROUP BY c.id
            ORDER BY c.created_at DESC
        """, (user["user_id"],))
        
        results = await cursor.fetchall()
        
        data_list = []
        for row in results:
            tags = []
            if row["tag_ids"]:
                tag_ids = row["tag_ids"].split(',')
                names = row["tag_names"].split(',')
                sources = row["tag_sources"].split(',')
                tags = [
                    {"tag_id": tid, "name": n, "source": s}
                    for tid, n, s in zip(tag_ids, names, sources)
                ]
                
            image_meta = None
            if row["file_path"]:
                image_meta = {
                    "width": row["width"],
                    "height": row["height"],
                    "file_size": row["file_size"],
                    "file_path": row["file_path"],
                    "thumbnail_path": row["thumbnail_path"]
                }
                
            data_list.append({
                **row,
                "tags": tags,
                "image_meta": image_meta
            })
            
        return JSONResponse(content=data_list, headers={"Cache-Control": "no-store"})
        
    except Exception as e:
        logger.error(f"Data fetch failed: {str(e)}")
        raise HTTPException(500, "Internal server error")

# 특정 데이터 조회
@app.get("/clipboard-data/{data_id}", response_model=ClipboardDataResponse)
async def get_single_clipboard_data(
    data_id: str, 
    user: Dict = Depends(get_current_user), 
    db=Depends(get_db)
):
    conn, cursor = db
    try:
        # 메타데이터 조회
        await cursor.execute("""
            SELECT c.*, 
                   GROUP_CONCAT(t.tag_id) AS tag_ids,
                   GROUP_CONCAT(t.name) AS tag_names,
                   GROUP_CONCAT(t.source) AS tag_sources,
                   im.width, im.height, im.file_size,
                   im.file_path, im.thumbnail_path
            FROM clipboard c
            LEFT JOIN data_tag dt ON c.id = dt.data_id
            LEFT JOIN tag t ON dt.tag_id = t.tag_id
            LEFT JOIN image_meta im ON c.id = im.data_id
            WHERE c.id = %s AND c.user_id = %s
            GROUP BY c.id
        """, (data_id, user["user_id"]))
        
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Data not found")

        # 태그 파싱
        tags = []
        if row["tag_ids"]:
            tag_ids = row["tag_ids"].split(',')
            names = row["tag_names"].split(',')
            sources = row["tag_sources"].split(',')
            tags = [
                {"tag_id": tid, "name": n, "source": s}
                for tid, n, s in zip(tag_ids, names, sources)
            ]
        original_url = None
        thumbnail_url = None
        if row["type"] == "img" and row.get("file_path"):
            # file_path: 절대경로 -> 파일명 추출 후 URL로 변환
            original_filename = os.path.basename(row["file_path"])
            thumbnail_filename = os.path.basename(row["thumbnail_path"]) if row.get("thumbnail_path") else None
            original_url = f"/images/original/{original_filename}"
            thumbnail_url = f"/images/thumbnail/{thumbnail_filename}" if thumbnail_filename else None
        # 응답 구성
        response_data = {
            **row,
            "tags": tags,
            "image_meta": {
                "width": row["width"],
                "height": row["height"],
                "file_size": row["file_size"],
                "original_url": original_url,
                "thumbnail_url": thumbnail_url,
            } if original_url else None
        }

        return response_data
        
    except Exception as e:
        logger.error(f"Data fetch failed: {str(e)}")
        raise HTTPException(500, "Internal server error")

# 이미지 파일 직접 다운로드 엔드포인트
@app.get("/clipboard-images/{data_id}")
async def get_image_file(
    data_id: str,
    user: Dict = Depends(get_current_user),
    db=Depends(get_db)
):
    conn, cursor = db
    try:
        await cursor.execute("""
            SELECT c.type, im.file_path 
            FROM clipboard c
            LEFT JOIN image_meta im ON c.id = im.data_id
            WHERE c.id = %s AND c.user_id = %s
        """, (data_id, user["user_id"]))
        
        row = await cursor.fetchone()
        if not row or row["type"] != "img":
            raise HTTPException(404, "Image not found")

        if not os.path.exists(row["file_path"]):
            raise HTTPException(404, "Image file not found")

        return FileResponse(
            row["file_path"],
            media_type="application/octet-stream",
            filename=os.path.basename(row["file_path"])
        )
        
    except Exception as e:
        logger.error(f"Image fetch failed: {str(e)}")
        raise HTTPException(500, "Internal server error")
    


@app.post("/search-text")
async def search_similar_images(
    data: TextSearchRequest,
    user: Dict = Depends(get_current_user),
    db=Depends(get_db)
):
    query = data.query
    user_id = user["user_id"]
    logger.info(f"CLIP 검색 요청 query: {query}")

    try:
        # KoCLIP 텍스트 임베딩
        vec = encode_text(query)
        vec = vec.tolist()

        # 이미 정의된 index 객체 사용
        result = index.query(
            vector=vec,
            top_k=10,
            include_metadata=True
        )

        # 유효한 ID만 필터링
        matches = result.get("matches", [])
        image_ids = [m.get("id") for m in matches if m.get("id") is not None]

        if not image_ids:
            return {"query": query, "results": []}

        # DB 연결
        conn, cursor = db
        placeholders = ",".join(["%s"] * len(image_ids))
        query_sql = f"""
            SELECT c.id
            FROM clipboard c
            WHERE c.user_id = %s
            AND c.id IN ({placeholders})
        """
        params = [user_id] + image_ids
        await cursor.execute(query_sql, params)
        rows = await cursor.fetchall()

        found_map = {row["id"]: True for row in rows}

        # 5. Pinecone 순서 기준으로 필터링 및 정렬
        ordered_ids = [id for id in image_ids if id in found_map]

        return {
            "query": query,
            "ids": ordered_ids
        }

    except Exception as e:
        logger.error(f"Text search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
# 에러 핸들러 (기존과 동일)
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

