# CalmWave Audio Benchmark API

API simples para avaliar qualidade de áudio (PESQ / STOI / SDR).

**Descrição**
- API em FastAPI que aceita dois arquivos áudio (`clean_file` e `denoised_file`) e retorna métricas de qualidade.

**Endpoints**
- `GET /test`: rota de saúde / teste — retorna JSON simples.
- `POST /benchmark`: recebe `multipart/form-data` com campos `clean_file` e `denoised_file` e retorna JSON com `pesq`, `stoi`, `sdr` e `status`.

**Rodando com Docker**
Build e run localmente:
```bash
docker build -t calmwave-api .
docker run -e PORT=8000 -p 8000:8000 calmwave-api
```

Depois acesse `http://localhost:8000/test`.

**Rodando localmente (sem Docker)**
1. Criar e ativar um ambiente virtual
```bash
# Linux / macOS
python3 -m venv venv && source venv/bin/activate
# Windows (PowerShell)
python -m venv venv; venv\Scripts\Activate.ps1
```
2. Instalar dependências e iniciar o servidor
```bash
pip install -r requirements.txt
uvicorn benchmark:app --host 0.0.0.0 --port 8000
```

**Exemplo de uso**
- Teste de saúde:
```bash
curl http://localhost:8000/test
```
- Envio de arquivos para avaliação:
```bash
curl -X POST "http://localhost:8000/benchmark" \
  -F "clean_file=@clean.wav" \
  -F "denoised_file=@denoised.wav"
```

**Arquivos principais**
- [benchmark.py](benchmark.py)
- [requirements.txt](requirements.txt)
- [Dockerfile](Dockerfile)
