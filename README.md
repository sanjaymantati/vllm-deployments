# vLLM + Prometheus + Grafana

A Docker Compose stack that serves an open-weight LLM through an **OpenAI-compatible API** using [vLLM](https://github.com/vllm-project/vllm), with **Prometheus** metrics, **Grafana** dashboards and **GPU telemetry** (NVIDIA DCGM).

- OpenAI-compatible endpoint (`/v1/chat/completions`, `/v1/models`, ...), API-key protected
- Pre-provisioned Grafana dashboard: queue depth, KV-cache usage, token throughput, TTFT, inter-token and end-to-end latency, GPU utilization and memory
- Everything configured from one `.env` file
- Services bind to `127.0.0.1` by default; put a TLS reverse proxy in front for remote access
- Tested end to end on an RTX 5050 Laptop (8 GB) with Docker Desktop on Windows/WSL2, vLLM `v0.32.0`

> **Status:** a solid single-node base. It is **not** production-grade out of the box: see [Production readiness](#production-readiness) for what is still missing (alerting, TLS, central logs, redundancy).

---

## Contents

1. [Architecture](#architecture)
2. [Requirements](#requirements)
3. [Quick start](#quick-start)
4. [Configuration reference](#configuration-reference)
5. [Choosing a model for your GPU](#choosing-a-model-for-your-gpu)
6. [Server setup (Ubuntu + NVIDIA)](#server-setup-ubuntu--nvidia)
7. [Windows (Docker Desktop + WSL2)](#windows-docker-desktop--wsl2)
8. [Verify the deployment](#verify-the-deployment)
9. [Using the API](#using-the-api)
10. [Monitoring](#monitoring)
11. [Securing and exposing it](#securing-and-exposing-it)
12. [Operations](#operations)
13. [Troubleshooting](#troubleshooting)
14. [Production readiness](#production-readiness)
15. [Repository layout](#repository-layout)

---

## Architecture

```
                 clients (curl, SDKs, Cursor, Continue, ...)
                              │  HTTPS + API key
                     ┌────────▼─────────┐
                     │ reverse proxy    │  (you add this: Caddy / nginx / Traefik)
                     └────────┬─────────┘
                              │ 127.0.0.1:8000
┌─────────────────────────────▼──────────────────────────────────────┐
│ docker compose                                                     │
│                                                                    │
│  ┌──────────┐ /metrics ┌────────────┐ query ┌─────────┐            │
│  │  vLLM    │◄─────────│ Prometheus │◄──────│ Grafana │            │
│  │  :8000   │          │   :9090    │       │  :3000  │            │
│  └────┬─────┘          └─────▲──────┘       └─────────┘            │
│       │ GPU                  │ :9400                               │
│       ▼                ┌─────┴──────┐                              │
│   NVIDIA GPU ─────────►│DCGM exporter│                             │
│                        └────────────┘                              │
└────────────────────────────────────────────────────────────────────┘
```

| Service | Image | Port (host) | Purpose |
|---|---|---|---|
| `vllm` | built from `vllm/Dockerfile` (`vllm/vllm-openai`) | `8000` | Model server, OpenAI-compatible API, `/metrics` |
| `dcgm-exporter` | `nvcr.io/nvidia/k8s/dcgm-exporter` | none (internal) | GPU utilization, memory, temperature |
| `prometheus` | `prom/prometheus` | `9090` | Metrics storage and queries |
| `grafana` | `grafana/grafana` | `3000` | Dashboards |

---

## Requirements

**Hardware**
- NVIDIA GPU with enough VRAM for your model (see [sizing](#choosing-a-model-for-your-gpu))
- Disk: ~20 GB for images plus the model weights (a 7B 4-bit model is ~5.5 GB; a 70B model is far larger)
- RAM: at least the container limit (`VLLM_MEM_LIMIT`, default 10 GB) plus headroom for the OS

**Software**
- Linux with a recent NVIDIA driver, **or** Windows 10/11 with Docker Desktop (WSL2 backend)
- Docker Engine 24+ with Compose v2 (`docker compose`, not `docker-compose`)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on Linux (not needed on Docker Desktop/WSL2; the Windows driver is enough)
- Newer GPUs (e.g. RTX 50-series / Blackwell) need a recent vLLM image. This repo pins `v0.32.0`, which worked on an RTX 5050. Older tags such as `v0.9.x` are reported to fail on RTX 50 cards.

---

## Quick start

```bash
git clone <your-repo-url> vllm-stack
cd vllm-stack

cp .env.example .env
# Generate secrets and paste them into .env:
openssl rand -hex 32     # -> VLLM_API_KEY
openssl rand -hex 16     # -> GRAFANA_PASSWORD

# Edit .env: at minimum set VLLM_API_KEY and GRAFANA_PASSWORD,
# and pick a MODEL that fits your GPU.

docker compose up -d --build
docker compose logs -f vllm     # wait for "Application startup complete"
```

The first start downloads the vLLM image (~10 GB) and the model weights, so it can take a while. Later starts reuse the `hf-cache` volume.

**Send one throwaway request after every (re)start.** With CUDA graphs disabled (`--enforce-eager`) the first request pays a one-time setup cost (~40 s was observed on an RTX 5050). Subsequent requests are fast.

---

## Configuration reference

All settings live in `.env` (copy from `.env.example`; `.env` is git-ignored).

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | Hugging Face model ID. Must fit in VRAM. |
| `SERVED_NAME` | `qwen2.5-coder-7b` | Model name clients put in the `"model"` field. |
| `MAX_MODEL_LEN` | `4096` | Max context (prompt + output) in tokens. Larger = more KV-cache memory. |
| `GPU_MEM_UTIL` | `0.80` | Fraction of VRAM vLLM may claim. See note below. |
| `TP_SIZE` | `1` | Tensor-parallel size. Set to the number of GPUs for multi-GPU. |
| `EXTRA_ARGS` | see file | Extra `vllm serve` flags (quantization, tool calling, batch size, ...). |
| `VLLM_API_KEY` | **required** | Bearer token for `/v1/*`. Compose refuses to start without it. |
| `GRAFANA_USER` / `GRAFANA_PASSWORD` | `admin` / **required** | Grafana login. |
| `HF_TOKEN` | empty | Only for gated Hugging Face models. |
| `BIND_ADDR` | `127.0.0.1` | Host interface for published ports. `0.0.0.0` exposes them on all interfaces. |
| `VLLM_CPUS` | `4` | CPU limit for the vLLM container (also sets `OMP_NUM_THREADS`). |
| `VLLM_MEM_LIMIT` | `10G` | RAM limit for the vLLM container. The container is killed (exit 137) if exceeded. |
| `VLLM_VERSION` | `v0.32.0` | vLLM image tag. Pin an exact version. |
| `PROM_RETENTION` | `15d` | How long Prometheus keeps data. |

**Notes**
- **`GPU_MEM_UTIL`** is a fraction of *total* VRAM, but vLLM requires that much to be *free* at startup. On a desktop or laptop the OS and display already hold some VRAM. On an 8 GB card with ~1 GB used, `0.88` failed with `Free memory ... is less than desired GPU memory utilization`; `0.80` worked. On a headless server, `0.90`-`0.95` is typical.
- **`EXTRA_ARGS` for tool calling:** clients that send `tools` (Cursor, Continue, agent frameworks) fail with HTTP 400 (`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`) unless you set both flags. The parser depends on the model family (`hermes` fits Qwen2.5; check the [vLLM tool-calling docs](https://docs.vllm.ai/) for others).
- **`--enforce-eager`** disables CUDA graphs to save VRAM at some cost in decode speed. On GPUs with more headroom, remove it.
- **`--quantization awq`** must match the model. Use it only with AWQ checkpoints; drop it for unquantized models.

---

## Choosing a model for your GPU

VRAM must hold **weights + KV cache + overhead**. Rule of thumb for weights: 16-bit ≈ 2 GB per billion parameters; 4-bit (AWQ/GPTQ) ≈ 0.55-0.7 GB per billion. KV cache grows with `MAX_MODEL_LEN` and concurrent requests (`--max-num-seqs`).

| GPU VRAM | Starting point | Notes |
|---|---|---|
| 8 GB | 7B in 4-bit (AWQ), `MAX_MODEL_LEN` 2048-4096 | **Verified** on RTX 5050 Laptop. Little KV headroom; 3B models allow longer context. |
| 12-16 GB | 7-8B in 16-bit, or 14B in 4-bit | Starting estimates, not verified here. |
| 24 GB | 14B in 16-bit, or ~32B in 4-bit | Starting estimates, not verified here. |
| 48-80 GB | 70B in 4-bit, or 30B+ in 16-bit | Starting estimates, not verified here. |
| Multi-GPU | Larger models with `TP_SIZE=<gpus>` | Needs fast interconnect for best results. |

Check how much KV cache you actually got in the startup log:

```bash
docker compose logs vllm | grep -i "KV cache"
```

If it reports fewer tokens than your `MAX_MODEL_LEN` times expected concurrency, lower `MAX_MODEL_LEN`, lower `--max-num-seqs`, or use a smaller/more quantized model.

**IDE agents (Cursor, etc.):** agent modes send tens of thousands of tokens of context and tool definitions. A 4096-token window will reject them (`maximum context length is 4096 tokens`). An 8 GB GPU is realistic for chat and small edits, not agent workflows.

---

## Server setup (Ubuntu + NVIDIA)

```bash
# 1. NVIDIA driver (skip if `nvidia-smi` already works)
sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install && sudo reboot

# 2. Docker Engine + Compose v2
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"      # log out/in afterwards

# 3. NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Confirm Docker can see the GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If step 4 prints your GPU table, continue with the [Quick start](#quick-start). Install steps change over time; confirm against the [official NVIDIA guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

**Firewall:** allow only SSH and your reverse proxy port (443). Do not open 8000, 9090 or 3000 to the internet (see [Securing](#securing-and-exposing-it)).

---

## Windows (Docker Desktop + WSL2)

1. Install the latest NVIDIA Windows driver. `nvidia-smi` must work in PowerShell.
2. Install Docker Desktop with the **WSL2 backend** (Settings → General → "Use the WSL 2 based engine"). You do **not** install the Container Toolkit or a Linux driver in WSL.
3. Update WSL: `wsl --update`.
4. Verify GPU passthrough:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```
5. Follow the [Quick start](#quick-start) in PowerShell. Use `copy .env.example .env`, and `curl.exe` instead of `curl` (PowerShell aliases `curl`).

**WSL memory cap:** Docker Desktop's total RAM limit comes from WSL (default: about half of system RAM). To change it, create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=16GB
```

then run `wsl --shutdown` and restart Docker Desktop.

---

## Verify the deployment

```bash
docker compose ps                         # vllm should be "Up (healthy)"
docker compose logs vllm | grep "startup complete"
curl -i http://localhost:8000/health      # HTTP 200 (no auth required)
curl http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

Generate text:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder-7b","messages":[{"role":"user","content":"Write hello world in Python"}],"max_tokens":50}'
```

(`"model"` must equal `SERVED_NAME`.)

Check the monitoring path:

- `http://localhost:8000/metrics` lists `vllm:` metrics
- `http://localhost:9090/targets` shows `vllm`, `dcgm` and `prometheus` as **UP**
- `http://localhost:3000` → Grafana → **vLLM Overview** (panels fill in after a few requests)
- `nvidia-smi` shows VRAM in use

---

## Using the API

**Python (OpenAI SDK)**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<VLLM_API_KEY>")
resp = client.chat.completions.create(
    model="qwen2.5-coder-7b",
    messages=[{"role": "user", "content": "Explain list comprehensions."}],
)
print(resp.choices[0].message.content)
```

**Editors and tools:** any OpenAI-compatible client works. Set the base URL to `https://<your-host>/v1`, the API key to `VLLM_API_KEY`, and the model name to `SERVED_NAME`.

- Cursor's "Override OpenAI Base URL" calls your endpoint from Cursor's servers, so it must be publicly reachable (hence TLS and a strong key are mandatory, not optional). Local-only setups will not work with it.
- Continue, Open WebUI, LiteLLM and similar tools call from your machine and can use a private/VPN address.

---

## Monitoring

The provisioned **vLLM Overview** dashboard:

| Panel | What to watch |
|---|---|
| Requests: running / waiting | `waiting` staying above 0 means requests are queueing: you are over capacity. |
| KV cache usage | Near 100% means new requests wait or get preempted. Lower context/concurrency or add VRAM. |
| Throughput (tokens/s) | Prompt vs. generation tokens per second. |
| Time to first token (p50/p95/p99) | User-perceived responsiveness. |
| Inter-token latency | Streaming speed per request. |
| End-to-end latency | Total request time. |
| Requests finished / s | Broken down by finish reason (`stop`, `length`, `abort`, `error`). |
| GPU utilization / memory | From DCGM. |

Notes:

- Latency spikes right after a restart are the cold start, not steady-state performance.
- vLLM renames metrics between releases. The dashboard queries tolerate a few known renames, but re-check panels after upgrading `VLLM_VERSION`.
- **DCGM on laptops/WSL2:** it worked on the tested Windows/WSL2 laptop, but may not on all setups. If the `dcgm-exporter` container fails, remove that service from `docker-compose.yml` and the `dcgm` job from `prometheus/prometheus.yml`; vLLM panels are unaffected.
- Duplicate `GPU 0` legend entries after restarts are cosmetic.

---

## Securing and exposing it

What the stack gives you by default:

- Ports are published on `127.0.0.1` only (`BIND_ADDR`).
- `/v1/*` requires `VLLM_API_KEY`.
- Compose refuses to start with an empty API key or Grafana password.

What it does **not** give you:

- **`/metrics` and `/health` are unauthenticated** (`--api-key` covers `/v1/*` only), and Prometheus has no auth. Never publish 8000, 9090 or 3000 directly to the internet, and do not tunnel port 8000 with ngrok or similar without filtering these paths.
- **No TLS.** API keys travel in headers; use HTTPS for anything beyond localhost.
- **One shared API key.** No per-user keys, quotas or audit trail.

### Recommended: TLS reverse proxy

Keep `BIND_ADDR=127.0.0.1` and expose only a reverse proxy. Example Caddyfile (automatic HTTPS; adapt and test before relying on it):

```caddyfile
llm.example.com {
    @internal path /metrics* /health*
    handle @internal {
        respond 404
    }
    handle {
        reverse_proxy 127.0.0.1:8000
    }
}
```

### Reaching Grafana and Prometheus remotely

Do not expose them. Use an SSH tunnel:

```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 user@your-server
```

Then open `http://localhost:3000` locally. Alternatively, put them behind a VPN (WireGuard, Tailscale) or an authenticated proxy.

### Checklist

- [ ] Strong random `VLLM_API_KEY` and `GRAFANA_PASSWORD`
- [ ] `.env` never committed (it is in `.gitignore`), file mode `600`
- [ ] Firewall allows only SSH and 443
- [ ] TLS in front of the API; `/metrics` and `/health` blocked at the proxy
- [ ] Rotate the API key if it was ever pasted into a chat, ticket or public tunnel

---

## Operations

```bash
docker compose ps                         # status
docker compose logs -f vllm               # follow logs
docker compose restart vllm               # restart one service
docker compose up -d --force-recreate vllm   # apply .env / compose changes
docker compose down                       # stop (keeps volumes)
docker compose down -v                    # stop AND delete volumes (model cache, metrics, Grafana data)
docker stats --no-stream vllm             # CPU / RAM against limits
```

**Change the model or settings:** edit `.env`, then `docker compose up -d --force-recreate vllm`, wait for startup, send a warm-up request.

**Upgrade vLLM:**

1. Change `VLLM_VERSION` in `.env` to the new tag.
2. `docker compose up -d --build vllm`
3. Check the logs for flag errors, and check the Grafana panels for renamed metrics.
4. Roll back by restoring the old tag.

**Resource limits:** set `VLLM_CPUS` and `VLLM_MEM_LIMIT` in `.env`. The model runs on the GPU, so the CPU is mostly idle (2-4 cores is typical); observed RAM use for a 7B AWQ model was ~6.6 GiB. A limit lower than real use gets the container killed (`exit 137`, `OOMKilled=true`).

**Backups:** the named volumes are `hf-cache` (re-downloadable), `prom-data` (metrics history) and `grafana-data` (users, ad-hoc dashboards). The provisioned dashboard and datasource are in git. To back up a volume:

```bash
docker run --rm -v vllm-stack_grafana-data:/data -v "$PWD":/backup alpine \
  tar czf /backup/grafana-data.tgz -C /data .
```

(The volume prefix is your Compose project name, by default the directory name.)

**Run on boot:** containers use `restart: unless-stopped`, so they return after a reboot once the Docker service starts.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Free memory on device ... is less than desired GPU memory utilization` | Other processes (desktop, display, other containers) hold VRAM | Lower `GPU_MEM_UTIL` (e.g. `0.80`); close GPU-heavy apps; check `nvidia-smi`. |
| `CUDA out of memory` at startup or on first request | Model plus KV cache exceed VRAM | Lower `MAX_MODEL_LEN` and `--max-num-seqs`; use a smaller or quantized model. |
| Container exits with code 137 / `OOMKilled=true` | RAM limit hit | Raise `VLLM_MEM_LIMIT`; on Windows raise the WSL limit in `.wslconfig`. |
| `could not select device driver "nvidia"` | Container Toolkit missing (Linux) | Install it and run `nvidia-ctk runtime configure`; restart Docker. |
| Image runs but GPU not found on Windows | Docker not using WSL2, or old WSL/driver | Enable the WSL2 engine, `wsl --update`, update the NVIDIA driver. |
| `socket hang up` / connection reset on the first request | Server still loading, or first-request cold start exceeded a client timeout | Wait for `Application startup complete`; use a longer client timeout; send a warm-up request. |
| First request takes 30-40 s | One-time setup with CUDA graphs disabled | Normal after each restart. |
| HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser` | Client sends tools; server flags missing | Add `--enable-auto-tool-choice --tool-call-parser <parser>` to `EXTRA_ARGS`. |
| HTTP 400 `maximum context length is N tokens` | Prompt longer than `MAX_MODEL_LEN` | Raise `MAX_MODEL_LEN` (needs KV-cache memory) or send less context. |
| HTTP 401 | Missing/wrong `Authorization: Bearer ...` | Use `VLLM_API_KEY` from `.env`. |
| HTTP 404 `model ... does not exist` | `"model"` doesn't match `SERVED_NAME` | Use the exact `SERVED_NAME`. |
| Compose fails with `set VLLM_API_KEY in .env` | Secret not set | Fill in `.env`. |
| `dcgm-exporter` restarts repeatedly | Unsupported on this GPU/platform (laptop, WSL2) | Remove the service and the `dcgm` scrape job. |
| Grafana panels empty | Prometheus target down, or metric renamed | Check `http://localhost:9090/targets`; compare names with `curl localhost:8000/metrics`. |
| Fails on a new GPU (e.g. Blackwell) with kernel/arch errors | vLLM image too old for the GPU | Use a recent `VLLM_VERSION`. |

For anything else, start with `docker compose logs --tail 100 vllm`. The root cause is usually in the first error, not the last.

---

## Production readiness

This repository is a **basic single-node stack**. Before serving real users, add:

| Area | Missing here | Suggested next step |
|---|---|---|
| Alerting | No alert rules | Prometheus alert rules + Alertmanager: target down, queue growing, KV cache > 90%, p95 latency, GPU temperature |
| TLS and auth | Shared key, no TLS | Reverse proxy with TLS; per-user keys; rate and request-size limits |
| Logs | Lost when containers are recreated | Central logging (Loki/ELK); per-key access logs with token counts |
| Redundancy | One GPU = single point of failure | Multiple replicas behind a load balancer; rolling upgrades |
| Warm-up | First request is slow after start | Automated warm-up request after startup |
| Capacity | Settings are educated guesses | Load-test (e.g. `vllm bench`, Locust) and set `MAX_MODEL_LEN` / `--max-num-seqs` from results; define SLOs |
| Secrets | Plain `.env` | Secret manager / Docker secrets |
| Hardware | Laptops sleep, throttle, reboot | Use a server-class GPU host for real traffic |

---

## Repository layout

```
.
├── docker-compose.yml                 # all services, limits, healthcheck
├── .env.example                       # copy to .env (git-ignored)
├── vllm/
│   └── Dockerfile                     # pins the vLLM image version
├── prometheus/
│   └── prometheus.yml                 # scrape jobs: vllm, dcgm, prometheus
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml # auto-registered datasource
    │   └── dashboards/dashboards.yml  # dashboard provider
    └── dashboards/
        └── vllm.json                  # "vLLM Overview" dashboard
```

## License

Add a license file before publishing (e.g. MIT or Apache-2.0). The services used here have their own licenses: vLLM (Apache-2.0), Prometheus (Apache-2.0), Grafana (AGPL-3.0), NVIDIA DCGM exporter (Apache-2.0). Model weights carry their own license terms; check the model card.
