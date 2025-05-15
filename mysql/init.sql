-- 환경 변수 대신 실제 값 사용 (Docker가 자동으로 치환하지 않음)
CREATE USER IF NOT EXISTS 'app_user'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON clipboard_db.* TO 'app_user'@'%';
FLUSH PRIVILEGES;