# Environment Decisions

Date: 2026-04-22

## WSL2 LoRA/Unsloth setup status

Status: BLOCKED (network + offline cache issue)

Completed:
- WSL2 verified and enabled (`wsl --status` shows default version 2)
- Ubuntu 24.04 installed and initialized
- Miniconda installed inside WSL2 (`~/miniconda3`)
- Connectivity diagnostics executed from WSL2
- Windows-side Linux package cache preparation attempted for offline conda install

Observed blockers:
- In WSL2, outbound TCP connections time out (ICMP ping works, TCP/443 does not)
- `curl` in WSL2 fails to reach `repo.anaconda.com`
- Offline conda create for `evollm-wsl` repeatedly fails due missing/offline fetch path for:
  - `ncurses-6.5-h7934f7d_0.conda`
- Even with manual archive placement, conda still tries to fetch remote URL and aborts in offline mode

Impact:
- Cannot complete requested environment installation sequence in WSL2:
  - `conda create -n evollm-wsl python=3.11`
  - `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
  - `pip install unsloth transformers accelerate datasets`
- Cannot run Unsloth smoke test in WSL2 yet
- Cannot export final WSL2 env specs yet

Next required action (host-level):
1. Restore outbound TCP connectivity from WSL2 NAT instance (firewall/network policy update on host).
2. After connectivity is restored, rerun environment install in WSL2 directly from network.

Verification command to re-check network when host fix is done:

```bash
wsl -d Ubuntu-24.04 -e bash -lc "python3 - <<'PY'
import socket
for host,port in [('repo.anaconda.com',443),('google.com',443)]:
    s=socket.socket(); s.settimeout(5)
    try:
        s.connect((host,port)); print(host,port,'OK')
    except Exception as e:
        print(host,port,'FAIL',e)
    finally:
        s.close()
PY"
```

Decision:
- Keep Windows environment for pre-0 tooling and tests.
- Keep WSL2 as required path for LoRA + Unsloth once WSL2 outbound TCP is restored.

## 2026-04-23 — Ollama dropped from pipeline

Ollama removed because qwen3:8b-base is not available in Ollama registry. Replaced by direct HuggingFace download (Qwen/Qwen3-8B-Base) with Unsloth as inference and fine-tuning backend. All logprobs for H_dezorg computed via Unsloth forward pass on frozen seed model.

## 2026-04-27 — Ollama removed from config

Ollama removed from config: inference_backend changed from ollama to unsloth. Generation parameters locked: temperature=0.0, do_sample=false, seed=42 for deterministic outputs.
