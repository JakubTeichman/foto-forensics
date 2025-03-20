FROM python:3.11

RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/

COPY frontend/ frontend/
WORKDIR /app/frontend
RUN npm install && npm run build

WORKDIR /app
RUN mkdir -p backend/static && cp -r frontend/build/* backend/static/

ENV FLASK_APP=backend/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

CMD ["flask", "run"]
