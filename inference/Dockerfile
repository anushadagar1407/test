FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY helpers.py .

EXPOSE 5000

ENTRYPOINT ["python", "inference.py"]