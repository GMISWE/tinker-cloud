# OpenTinker Miles Training API

Tinker -> tinke-miles -> miles for e2e training

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn training.api:app --reload --port 8000
```

## Docker

```bash
cd docker
docker build -t opentinker/miles-training:latest .
```
