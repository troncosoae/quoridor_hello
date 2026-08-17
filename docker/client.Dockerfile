FROM python:3.13-slim

WORKDIR /app
COPY quoridor ./quoridor

ENTRYPOINT ["python", "-m", "quoridor.play_remote"]
