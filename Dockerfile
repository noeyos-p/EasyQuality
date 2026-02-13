FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치 (PDF 처리, psycopg2 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    curl \
    pkg-config \
    libcairo2-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# torch 설치 (ARM/x86 자동 감지)
RUN arch=$(uname -m) && \
    if [ "$arch" = "aarch64" ]; then \
        pip install --no-cache-dir torch torchvision; \
    else \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 복사
COPY main.py .
COPY backend/ ./backend/
COPY fonts/ ./fonts/

EXPOSE 8000

CMD ["python", "main.py"]
