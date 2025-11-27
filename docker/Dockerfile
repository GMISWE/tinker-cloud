FROM us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/miles_dev-202511120a:latest

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY training ./training

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

CMD ["python3", "-m", "training"]
