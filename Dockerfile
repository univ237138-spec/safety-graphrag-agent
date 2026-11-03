# 데모 앱 이미지 (키 없이 예시 재생·목록 조회). 실행: docker compose up  → http://localhost:8501
# 자유 질문까지(본인 키): docker run --rm -p 127.0.0.1:8501:8501 -v "$PWD/.env:/app/.env:ro" safety-graphrag-demo
#   (.env에 OPENAI_API_KEY와 DEMO_LIVE=1. 키는 이미지에 넣지 않는다 — .dockerignore)
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
