
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY best_model.joblib .

EXPOSE 8000

CMD ["sh", "-c", "fastapi run api.py --host 0.0.0.0 --port ${PORT:-8000}"]
