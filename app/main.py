from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager
from typing import Optional, Dict, Literal
from datetime import datetime, timedelta
import aiomysql
import uuid
import bcrypt
import jwt
import logging
from fastapi_limiter.depends import RateLimiter
from fastapi_limiter import FastAPILimiter
from redis.asyncio import Redis

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

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# 데이터 모델 (기존과 동일)
class UserSignupRequest(BaseModel):
    user_id: str = Field(..., min_length=4, max_length=50)
    password: str = Field(..., min_length=8)

class UserLoginRequest(UserSignupRequest):
    pass

class ItemCreate(BaseModel):
    content: str
    type: str = Field(..., pattern="^(img|txt)$")
    format: str

class ItemResponse(ItemCreate):
    id: str
    created_at: int

# 데이터베이스 초기화 함수
async def initialize_database():
    DROP_TABLES = """
        DROP TABLE IF EXISTS data_tag;
        DROP TABLE IF EXISTS tag;
        DROP TABLE IF EXISTS image_meta;
        DROP TABLE IF EXISTS clipboard;
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
            type ENUM('img', 'txt') NOT NULL,
            format VARCHAR(50) NOT NULL,
            content VARCHAR(255) NOT NULL,
            created_at INTEGER NOT NULL,
            shared ENUM('cloud', 'local') NOT NULL
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
            sync_status VARCHAR(20) DEFAULT 'pending',
            UNIQUE (name, source)
        );
        
        CREATE TABLE IF NOT EXISTS data_tag (
            data_id VARCHAR(36) NOT NULL,
            tag_id VARCHAR(36) NOT NULL,
            PRIMARY KEY (data_id, tag_id),
            FOREIGN KEY (data_id) REFERENCES clipboard(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tag(tag_id) ON DELETE CASCADE
        );
    """

    async with app_state.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                await cursor.execute(DROP_TABLES)
                await cursor.execute(CREATE_TABLES)
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                logger.error(f"Database initialization failed: {str(e)}")
                # 테이블이 이미 존재할 경우 계속 진행
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
            charset="utf8mb4"
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

# 의존성 주입
async def get_db():
    async with app_state.db_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            yield conn, cursor

security = HTTPBearer(auto_error=False)

# JWT 유틸리티 (기존과 동일)
def create_access_token(user_id: str) -> str:
    expire = datetime.now(datetime.timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode(
        {"sub": user_id, "exp": expire},
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

# 엔드포인트 (기존과 동일하되 모든 execute에 await 추가)
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
            (user.user_id, hashed_pw, int(datetime.now(datetime.timezone.utc).timestamp()))
        )
        await conn.commit()
        return {"message": "User created"}

    except Exception as e:
        await conn.rollback()
        logger.error("Signup failed", exc_info=e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/login")
async def login(user: UserLoginRequest, db=Depends(get_db)):
    conn, cursor = db
    try:
        await cursor.execute(
            "SELECT password FROM user WHERE user_id = %s",
            (user.user_id,)
        )
        result = await cursor.fetchone()
        
        if not result or not bcrypt.checkpw(user.password.encode(), result["password"].encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED)

        token = create_access_token(user.user_id)
        return {"access_token": token, "token_type": "bearer"}

    except Exception as e:
        logger.error("Login failed", exc_info=e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/items", response_model=ItemResponse)
async def create_item(item: ItemCreate, user: Dict = Depends(get_current_user), db=Depends(get_db)):
    conn, cursor = db
    try:
        item_id = str(uuid.uuid4())
        created_at = int(datetime.now(datetime.timezone.utc).timestamp())

        await cursor.execute(
            "INSERT INTO clipboard (id, content, type, format, created_at, shared) VALUES (%s, %s, %s, %s, %s, 'local')",
            (item_id, item.content, item.type, item.format, created_at)
        )
        await conn.commit()

        return {**item.model_dump(), "id": item_id, "created_at": created_at}

    except Exception as e:
        await conn.rollback()
        logger.error("Item creation failed", exc_info=e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)

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