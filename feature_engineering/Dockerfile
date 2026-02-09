FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_engineering.py .
COPY helpers.py .

ENTRYPOINT ["python", "feature_engineering.py"]