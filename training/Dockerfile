FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# data/ (entrada) e model/ (saída) chegam por volume — ver docker-compose.yml
CMD ["python", "train.py"]
