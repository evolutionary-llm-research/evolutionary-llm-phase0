"""Pre-0 environment smoke test for Evolutionary LLM Research."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _try_import_subprocess(module_name: str) -> tuple[bool, str]:
    cmd = [sys.executable, "-c", f"import {module_name}; print('ok')"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, "ok"

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    message = stderr if stderr else stdout
    return False, f"exit_code={proc.returncode}; {message}"


def run_smoke_test() -> int:
    required_modules = [
        "torch",
        "accelerate",
        "peft",
        "trl",
        "datasets",
        "sentence_transformers",
        "scipy",
        "yaml",
    ]
    optional_modules = ["unsloth"]

    results: dict[str, dict[str, str | bool]] = {}

    for module in required_modules:
        ok, message = _try_import_subprocess(module)
        results[module] = {"ok": ok, "message": message, "required": True}

    for module in optional_modules:
        # Unsloth can terminate the interpreter on some Windows builds.
        ok, message = _try_import_subprocess(module)
        results[module] = {"ok": ok, "message": message, "required": False}

    torch_meta = {
        "version": "unknown",
        "cuda_runtime": None,
        "cuda_available": False,
        "device_name": None,
    }
    try:
        import torch

        torch_meta["version"] = torch.__version__
        torch_meta["cuda_runtime"] = torch.version.cuda
        torch_meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch_meta["cuda_available"]:
            torch_meta["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        torch_meta["error"] = f"{type(exc).__name__}: {exc}"

    required_ok = all(bool(results[name]["ok"]) for name in required_modules)
    unsloth_ok = bool(results["unsloth"]["ok"])
    cuda_ok = bool(torch_meta["cuda_available"])

    status = "PASS"
    exit_code = 0

    if not required_ok or not cuda_ok:
        status = "FAIL"
        exit_code = 1
    elif not unsloth_ok and platform.system().lower().startswith("win"):
        status = "PASS_WITH_WARNINGS"
        exit_code = 0
    elif not unsloth_ok:
        status = "FAIL"
        exit_code = 1

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version,
        },
        "torch": torch_meta,
        "modules": results,
    }

    report_dir = Path("experiments") / "pre0_environment"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "pre0_smoke_test_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"PRE0_STATUS={status}")
    print(f"REPORT_PATH={report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())
