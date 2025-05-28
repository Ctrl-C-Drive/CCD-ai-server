FROM python:3.10-slim

# 시스템 종속성 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    dos2unix \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY wait-for-it.sh /app/wait-for-it.sh
RUN dos2unix ./wait-for-it.sh && chmod +x ./wait-for-it.sh
COPY . /app

# FastAPI 서버 실행
CMD ["./wait-for-it.sh", "mysql-db:3306", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]