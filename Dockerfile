FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/bot.py /app/bot.py
COPY app/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN useradd -m botuser
USER botuser

ENV PYTHONUNBUFFERED=1
CMD ["/app/entrypoint.sh"]
