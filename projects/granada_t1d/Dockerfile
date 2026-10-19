FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_PUBLIC_DEPLOYMENT=1 APP_ENABLE_RESEARCH=0
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils && rm -rf /var/lib/apt/lists/*
COPY requirements/app.txt requirements/app.txt
RUN pip install --no-cache-dir -r requirements/app.txt
COPY src src
COPY Phase_5_Application/app.py Phase_5_Application/app.py
COPY app.py app.py
COPY .streamlit/config.toml .streamlit/config.toml
RUN useradd --create-home appuser
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
