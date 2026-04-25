FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV ENABLE_WEB_INTERFACE=true
EXPOSE 7860
HEALTHCHECK CMD curl -f http://localhost:7860/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

