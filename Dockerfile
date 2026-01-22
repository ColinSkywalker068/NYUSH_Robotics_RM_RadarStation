FROM python:3.9.13-slim
WORKDIR /app
COPY . .
RUN python -m pip install -r requirements.txt