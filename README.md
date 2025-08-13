### How to run locally

```bash
pip install -r requirements-arm.txt
python main.py
```

### How to run locally by vitural env
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements-arm.txt
python main.py
```

### How to run via Docker
```bash
docker build --platform linux/x86_64 -t sizestrainmodel .
docker run -p 80:80 sizestrainmodel
```

### How to push image changes to dockerhub
```bash
docker buildx build --platform linux/amd64 -t luisrodriguesds/sizestrainmodel:latest --push .
```
