#!/bin/bash
# ============
# EasyQuality S3 배포 스크립트
# ============

# 변수 설정
DOCKER_USERNAME="noeyos"
PROJECT_NAME="easyquality"
EC2_IP="15.165.141.102"
EC2_USER="ubuntu"
SSH_KEY="$HOME/Downloads/aws-noeyos.pem"

# ============
# 1. Docker 이미지 빌드
# ============
echo "🔨 Building Docker images..."

# Backend 빌드
docker build -t ${DOCKER_USERNAME}/${PROJECT_NAME}-backend:latest .

# Frontend 빌드
cd frontend
docker build -t ${DOCKER_USERNAME}/${PROJECT_NAME}-frontend:latest .
cd ..

# ============
# 2. Docker Hub에 Push
# ============
echo "📤 Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-backend:latest
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-frontend:latest

# ============
# 3. EC2 서버에 배포 파일 전송
# ============
echo "📦 Copying files to EC2..."
scp -i "${SSH_KEY}" docker-compose.prod.yml ${EC2_USER}@${EC2_IP}:~/app/
scp -i "${SSH_KEY}" .env.prod ${EC2_USER}@${EC2_IP}:~/app/

# ============
# 4. EC2 서버에서 배포 실행
# ============
echo "🚀 Deploying on EC2..."
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} << 'EOF'
  cd ~/app

  # 기존 컨테이너 중지 및 제거
  docker compose -f docker-compose.prod.yml down

  # 최신 이미지 Pull
  docker pull noeyos/easyquality-backend:latest
  docker pull noeyos/easyquality-frontend:latest

  # 컨테이너 실행
  docker compose -f docker-compose.prod.yml --env-file .env.prod up -d

  # 상태 확인
  echo "📊 Container Status:"
  docker ps
EOF

echo "✅ Deployment completed!"
echo "🌐 Access: http://${EC2_IP}"
