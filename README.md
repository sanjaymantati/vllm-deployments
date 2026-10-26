# vLLM + Prometheus + Grafana

## Requirements
- Docker + Compose v2
- NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

## Run
```bash
# 1. Edit .env: set VLLM_API_KEY, GRAFANA_PASSWORD, HF_TOKEN (if gated model), MODEL
docker compose up -d --build
docker compose logs -f vllm      # wait for "Application startup complete"
```

## Endpoints
| Service    | URL                      |
|------------|--------------------------|
| vLLM API   | http://localhost:8000/v1 |
| Prometheus | http://localhost:9090    |
| Grafana    | http://localhost:3000 (dashboard: "vLLM Overview") |

## Test
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder-7b","messages":[{"role":"user","content":"hi"}]}'
```

## Notes
- `/metrics` and `/health` are unauthenticated; `--api-key` covers `/v1/*` only. Do not expose 8000, 9090 or 3000 publicly without a reverse proxy/firewall.
- vLLM metric names change between releases. Version is pinned in `.env`; re-check the dashboard when upgrading.
- Multi-GPU: set `TP_SIZE` to the GPU count.


NGROK COMMAND FOR CURSOR IDE ACCESS
```
ngrok http 8000 --host-header="localhost:8000"
```