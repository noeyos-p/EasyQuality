#!/bin/bash

# 🚀 빌드 & Push 스크립트
# 로컬에서 실행

set -e  # 에러 발생 시 중단

echo "📁 프로젝트 디렉토리로 이동..."
cd /Users/soyeon/Desktop/easy_quality_S3

echo ""
echo "🔨 Backend 이미지 빌드 중..."
docker build -t noeyos/easyquality-backend:latest .

echo ""
echo "🔨 Frontend 이미지 빌드 중..."
docker build -t noeyos/easyquality-frontend:latest -f frontend/Dockerfile frontend/

echo ""
echo "📤 Docker Hub에 Push 중..."
docker push noeyos/easyquality-backend:latest
docker push noeyos/easyquality-frontend:latest

echo ""
echo "✅ 빌드 & Push 완료!"
echo ""
echo "📋 다음 단계:"
echo "  1. EC2 서버에 접속"
echo "  2. 다음 명령어 실행:"
echo "     docker pull noeyos/easyquality-backend:latest"
echo "     docker pull noeyos/easyquality-frontend:latest"
echo "     docker-compose -f docker-compose.prod.yml down"
echo "     docker-compose -f docker-compose.prod.yml up -d"
