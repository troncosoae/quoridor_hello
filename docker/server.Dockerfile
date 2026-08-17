FROM python:3.13-slim

WORKDIR /app
COPY quoridor ./quoridor

EXPOSE 8765
CMD ["python", "-m", "quoridor.server", "--host", "0.0.0.0", "--port", "8765"]
