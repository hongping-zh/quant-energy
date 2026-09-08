#!/usr/bin/env python3
"""Generate notebooks/ecocompute-measure.ipynb.

The notebook is a no-Docker, free-Colab-T4 path to the same NVML measurement the
MLCube container performs. Keeping it generated from this script keeps the code
cells reviewable as plain Python.
"""
import json
import pathlib

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md(r"""
# EcoCompute: measure LLM inference energy on your GPU

**No Docker, ~20–35 minutes, a free Colab T4 is enough.**

This notebook runs the same NVML measurement as the
[EcoCompute MLCube container](https://github.com/hongping-zh/ecocompute-mlcube):
it measures a quantized model and its own FP16 baseline, in randomized order with a
forced cooldown before each arm, and reports `vs_fp16_energy_pct` — the change in
energy per token.

| | |
|---|---|
| **What you need** | A Colab runtime with a GPU (Runtime → Change runtime type → T4) |
| **Time** | ~20–35 min for both arms on a ≤3B model |
| **Output** | An `energy.json` in the container's schema, plus text you can paste into an issue |
| **Method** | NVML GPU-package power sampling, batch 1, 256 tokens, 2 warmup + 10 decode iterations |

**Read this before you trust the number it prints:**

- Power here is **GPU-package telemetry**, not wall AC. It excludes CPU, RAM, PSU losses and cooling.
- Energy is the **trapezoidal integral of sampled power**. In our
  [45-run RTX 4090 rerun](https://doi.org/10.5281/zenodo.22295184), the GPU's hardware
  energy counter read on average **14.9% higher** (range 2.5–31.8%) than integrating the same
  traces. Treat these figures as accurate to roughly that order, not to three digits.
- One pass through this notebook is **n=1**. The optional last cell repeats both arms 5×.
- This is **not** a certified benchmark result and not an MLPerf result.

> **Why does this exist?** The [EcoCompute dataset](https://doi.org/10.5281/zenodo.19647290)
> shows quantization does *not* always save energy — the sign flips with model size, GPU
> architecture and runtime. This notebook is how you check that on a GPU we do not have.
""")

md(r"""
## Step 1 · Install dependencies

Run once, ~2–4 minutes. Colab will likely ask you to restart the runtime after torch is
replaced — **restart, then run this cell again** and continue from Step 2.

This installs torch from the **CUDA 12.1** index rather than the default build. To be clear
about why: we have *not* measured a problem with newer CUDA builds. This pin exists so that
every contributed measurement comes from the same stack as the container, and because we
would rather not debug a bitsandbytes kernel difference after the fact. If the pin fails on
your runtime, the notebook still runs — it records whichever CUDA version it actually got.
""")

code(r"""
# Install torch from the cu121 index, then the rest.
!pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --quiet
!pip install bitsandbytes transformers accelerate pynvml sentencepiece --quiet

import torch

print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  cuda available: {torch.cuda.is_available()}")
if torch.version.cuda != "12.1":
    print(
        f"\nNote: torch CUDA is {torch.version.cuda}, not the pinned 12.1.\n"
        "  If Colab asked you to restart the runtime, restart and re-run this cell.\n"
        "  Otherwise you can continue: the actual version is recorded in the report."
    )

import pynvml

pynvml.nvmlInit()
_h = pynvml.nvmlDeviceGetHandleByIndex(0)
_name = pynvml.nvmlDeviceGetName(_h)
if isinstance(_name, bytes):
    _name = _name.decode()
try:
    _mw = pynvml.nvmlDeviceGetPowerUsage(_h)
    print(f"GPU: {_name} — NVML power telemetry OK ({_mw / 1000:.1f} W idle)")
except Exception as exc:
    print(f"GPU: {_name} — NVML power telemetry NOT available: {exc}")
    print("  This GPU cannot produce measured energy figures. Try a different runtime.")
pynvml.nvmlShutdown()
""")

md(r"""
## Step 2 · Choose model and precision

Edit the two variables, then run the cell. It refuses to start if the model will not fit,
rather than letting you wait ten minutes for an out-of-memory error.

`vs_fp16_energy_pct` requires **both** arms, so the FP16 baseline has to fit as well. That is
the real constraint on a 16 GB T4 — a 7B model loads fine in NF4 but its FP16 baseline does not:

| Precision | Quantized arm alone | With the FP16 baseline (needed for `vs_fp16`) |
|---|---|---|
| `NF4` | up to ~7B | **up to ~3B** |
| `INT8` | up to ~7B | **up to ~3B** |
| `FP16` | up to ~3B | n/a (it *is* the baseline) |

To measure a 7B model on a T4 anyway, set `RUN_FP16_BASELINE = False`. You then get an
energy-per-token figure with no `vs_fp16` — still useful, but it is not a comparison.
""")

code(r'''
# ─────────────────────────────────────────────────────────────
#  EDIT THESE, then run this cell
# ─────────────────────────────────────────────────────────────

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # any HuggingFace causal-LM id
PRECISION = "NF4"                                   # NF4 | INT8 | FP16

RUN_FP16_BASELINE = True    # False = quantized arm only, no vs_fp16
PARAMS_B_OVERRIDE = None    # set a float if the parameter count is not in the model name

# ─────────────────────────────────────────────────────────────

import re

import pynvml

assert PRECISION in ("NF4", "INT8", "FP16"), f"Invalid precision: {PRECISION}"

if PARAMS_B_OVERRIDE is not None:
    PARAMS_B = float(PARAMS_B_OVERRIDE)
else:
    _m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", MODEL_NAME.rsplit("/", 1)[-1])
    if _m is None:
        raise ValueError(
            f"Cannot read a parameter count out of {MODEL_NAME!r} "
            "(names like 'Phi-3-mini' do not contain one).\n"
            "Set PARAMS_B_OVERRIDE above — e.g. PARAMS_B_OVERRIDE = 3.8 — and re-run.\n"
            "Guessing here would put a wrong model size into the dataset."
        )
    PARAMS_B = float(_m.group(1))

# ── Will it fit? ──
_BYTES_PER_PARAM = {"FP16": 2.0, "INT8": 1.0, "NF4": 0.55}
_RUNTIME_OVERHEAD_GB = 1.2   # CUDA context, activations, KV cache at batch 1 / 256 tokens


def _need_gb(precision):
    return PARAMS_B * _BYTES_PER_PARAM[precision] * 1.05 + _RUNTIME_OVERHEAD_GB


pynvml.nvmlInit()
_mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
pynvml.nvmlShutdown()
VRAM_GB = _mem.total / 1e9

_arms = [PRECISION] + (["FP16"] if RUN_FP16_BASELINE and PRECISION != "FP16" else [])
print(f"Config: {MODEL_NAME} ({PARAMS_B}B) + {PRECISION}")
print(f"GPU memory: {VRAM_GB:.1f} GB total")
for _arm in _arms:
    print(f"  {_arm:<5s} needs about {_need_gb(_arm):.1f} GB")

_too_big = [a for a in _arms if _need_gb(a) > VRAM_GB * 0.95]
if _too_big:
    _hint = (
        "Set RUN_FP16_BASELINE = False to measure the quantized arm only (no vs_fp16),\n"
        "or pick a smaller model."
        if _too_big == ["FP16"]
        else "Pick a smaller model, or a runtime with more GPU memory."
    )
    raise MemoryError(
        f"{' and '.join(_too_big)} will not fit in {VRAM_GB:.1f} GB.\n{_hint}"
    )
print("\nFits. Run Step 3.")
''')

md(r"""
## Step 3 · Measurement engine

The sampler and the measurement loop, extracted from
[ecocompute-mlcube/entrypoint.py](https://github.com/hongping-zh/ecocompute-mlcube/blob/main/entrypoint.py)
with two additions: it records the sampling rate it *actually* achieved rather than the one
it asked for, and it waits for the GPU to return to idle before each arm. Run it, no edits.
""")

code(r'''
# ── EcoCompute measurement engine (from entrypoint.py, CC BY 4.0) ──

import datetime
import gc
import json
import platform
import re
import sys
import threading
import time

# ── Architecture detection ──

ARCH_PATTERNS = [
    (r"\b(b100|b200|gb\d{3}|rtx\s*50\d0)\b", "blackwell"),
    (r"\b(h100|h200|h800|gh200)\b", "hopper"),
    (r"\b(l4|l40s?|ada)\b|\brtx\s*(40\d0|(?:2000|4000|5000|6000)\s*ada)\b", "ada"),
    (r"\b(a100|a800|a10g?|a16|a2|a30|a40)\b|\brtx\s*(a\d{4}|30\d0)\b", "ampere"),
    (r"\b(t4|t400|t600|t1000)\b|\b(quadro|titan)\s*rtx\b|\brtx\s*20\d0\b", "turing"),
    (r"\bv100\b|\btitan\s*v\b", "volta"),
]


def detect_arch(gpu_name):
    if not gpu_name:
        return None
    s = re.sub(r"[-_]+", " ", str(gpu_name).lower())
    s = re.sub(r"\b(nvidia|geforce|tesla)\b", " ", s)
    for pattern, arch in ARCH_PATTERNS:
        if re.search(pattern, s):
            return arch
    return None


# ── PowerSampler: NVML power sampling in a background thread ──


class PowerSampler(threading.Thread):
    """Samples GPU power via NVML at a requested rate; integrates energy (trapezoid)."""

    def __init__(self, handle, hz=100):
        super().__init__(daemon=True)
        self._pynvml = sys.modules["pynvml"]
        self.handle = handle
        self.requested_hz = hz
        self.period = 1.0 / hz
        self.samples = []          # (t_seconds, watts)
        self.dropped = 0
        self.error = None
        self._stop_evt = threading.Event()

    def run(self):
        t0 = time.time()
        consecutive = 0
        while not self._stop_evt.is_set():
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append((time.time() - t0, mw / 1000.0))
                consecutive = 0
            except Exception as exc:
                self.dropped += 1
                consecutive += 1
                if consecutive >= 5 and not self.samples:
                    self.error = f"NVML power telemetry unavailable: {exc}"
                    break
            time.sleep(self.period)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)

    def _span(self):
        return self.samples[-1][0] - self.samples[0][0] if len(self.samples) >= 2 else 0.0

    def energy_joules(self):
        j = 0.0
        for (t1, w1), (t2, w2) in zip(self.samples, self.samples[1:]):
            j += (w1 + w2) / 2.0 * (t2 - t1)
        return j

    def avg_watts(self):
        return sum(w for _, w in self.samples) / len(self.samples) if self.samples else 0.0

    def achieved_hz(self):
        """Samples actually taken per second. The GIL and NVML latency put this below the request."""
        span = self._span()
        return (len(self.samples) - 1) / span if span > 0 else 0.0

    def nvml_update_hz(self):
        """How often the reading actually changed — NVML's own refresh rate, an upper bound
        on the time resolution of the integral no matter how fast we poll."""
        span = self._span()
        if span <= 0:
            return 0.0
        changes = sum(1 for (_, a), (_, b) in zip(self.samples, self.samples[1:]) if a != b)
        return changes / span


# ── Cooldown: return to idle before measuring, instead of assuming a fixed sleep is enough ──


def read_power_temp(handle):
    pynvml = sys.modules["pynvml"]
    watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        temp = None
    return watts, temp


def cooldown(handle, idle_watts, idle_temp_c, max_wait_s=240, poll_s=2.0,
             watt_margin=6.0, temp_margin=5.0, verbose=True):
    """Wait until power and temperature come back near the idle reference.

    Returns a record of what happened, including whether it actually converged — a fixed
    sleep does not, and the difference belongs in the report rather than in a footnote.
    """
    t0 = time.time()
    watts, temp = read_power_temp(handle)
    while time.time() - t0 < max_wait_s:
        watts, temp = read_power_temp(handle)
        power_ok = watts <= idle_watts + watt_margin
        temp_ok = idle_temp_c is None or temp is None or temp <= idle_temp_c + temp_margin
        if power_ok and temp_ok:
            break
        time.sleep(poll_s)
    waited = time.time() - t0
    converged = watts <= idle_watts + watt_margin and (
        idle_temp_c is None or temp is None or temp <= idle_temp_c + temp_margin
    )
    if verbose:
        state = "idle" if converged else "NOT back to idle (timed out)"
        print(f"  cooldown {waited:.0f}s -> {watts:.1f} W, {temp if temp is not None else '?'} C [{state}]")
    return {
        "waited_seconds": round(waited, 1),
        "start_watts": round(watts, 1),
        "start_temp_c": temp,
        "converged": bool(converged),
        "target_watts": round(idle_watts + watt_margin, 1),
        "target_temp_c": None if idle_temp_c is None else idle_temp_c + temp_margin,
    }


def measure_idle_reference(handle, seconds=15):
    """Idle power/temperature of this runtime, measured before anything is loaded."""
    watts, temps = [], []
    for _ in range(int(seconds / 0.5)):
        w, t = read_power_temp(handle)
        watts.append(w)
        if t is not None:
            temps.append(t)
        time.sleep(0.5)
    return min(watts), (min(temps) if temps else None)


# ── Quantization config ──


def _quant_config(precision):
    import torch
    from transformers import BitsAndBytesConfig

    if precision == "NF4":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    if precision == "INT8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ── Software version collection ──


def _pkg_version(name):
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def collect_software():
    versions = {
        name: _pkg_version(name)
        for name in ("torch", "transformers", "bitsandbytes", "accelerate", "nvidia-ml-py", "sentencepiece")
    }
    sw = {"python": platform.python_version(), "packages": {k: v for k, v in versions.items() if v}}
    try:
        import torch

        sw["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            drv = pynvml.nvmlSystemGetDriverVersion()
            sw["nvidia_driver"] = drv.decode() if isinstance(drv, bytes) else str(drv)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return sw


# ── Core measurement function ──


def measure_once(model_name, precision, idle_watts, idle_temp_c, batch_size=1, tokens=256,
                 iterations=10, warmup=2, hz=100):
    """Cool down, load, quantize, warm up, then measure energy over `iterations` decode runs."""
    import pynvml
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        pynvml.nvmlDeviceGetPowerUsage(handle)  # fail fast if there is no telemetry
    except Exception as exc:
        pynvml.nvmlShutdown()
        raise RuntimeError(f"NVML power telemetry not available: {exc}") from exc

    model = None
    try:
        cool = cooldown(handle, idle_watts, idle_temp_c)

        print(f"  loading {model_name} ({precision})...")
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        kwargs = {"torch_dtype": torch.float16, "device_map": "cuda"}
        qc = _quant_config(precision)
        if qc is not None:
            kwargs["quantization_config"] = qc
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()

        prompt = ["Explain in detail how large language models work."] * batch_size
        enc = tok(prompt, return_tensors="pt", padding=True).to("cuda")
        prompt_tokens = int(enc["input_ids"].shape[1])
        # min_new_tokens == max_new_tokens forces exactly `tokens` decode steps, so an early
        # EOS cannot shorten a run and inflate the saving. This is the notebook's equivalent
        # of llama.cpp's --ignore-eos.
        gen_kwargs = dict(
            max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )

        print(f"  warming up ({warmup} runs)...")
        with torch.no_grad():
            for _ in range(warmup):
                model.generate(**enc, **gen_kwargs)
        torch.cuda.synchronize()

        print(f"  measuring ({iterations} decode iterations, {hz} Hz requested)...")
        sampler = PowerSampler(handle, hz=hz)
        sampler.start()
        t0 = time.time()
        total_new = 0
        with torch.no_grad():
            for i in range(iterations):
                out = model.generate(**enc, **gen_kwargs)
                total_new += (out.shape[1] - prompt_tokens) * batch_size
                if (i + 1) % 5 == 0:
                    print(f"    iteration {i + 1}/{iterations}")
        torch.cuda.synchronize()
        wall = time.time() - t0
        sampler.stop()

        if sampler.error or len(sampler.samples) < 2:
            raise RuntimeError(sampler.error or "NVML returned too few power samples")

        joules = sampler.energy_joules()
        return {
            "gpu_name": gpu_name,
            "precision": precision,
            "total_energy_joules": round(joules, 3),
            "tokens_generated": total_new,
            "energy_per_token_mj": round(joules / total_new * 1000.0, 3),
            "avg_power_watts": round(sampler.avg_watts(), 1),
            "throughput_tokens_per_s": round(total_new / wall, 1),
            "wall_seconds": round(wall, 3),
            "prompt_tokens": prompt_tokens,
            "dropped_samples": sampler.dropped,
            "n_samples": len(sampler.samples),
            "sample_rate_hz_requested": sampler.requested_hz,
            "sample_rate_hz_achieved": round(sampler.achieved_hz(), 1),
            "nvml_update_hz_effective": round(sampler.nvml_update_hz(), 1),
            "cooldown_before_run": cool,
        }
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


print("Measurement engine ready.")
''')

md(r"""
## Step 4 · Run the measurement

Two arms — your precision and its FP16 baseline — in **randomized order**, each preceded by a
cooldown that waits for the card to come back to idle. Order and thermal state are the two
things that quietly bias a back-to-back comparison, and we published a
[whole post](https://quantenergy.tech/blog/rerunning-a-number-you-do-not-trust.html) about
getting them wrong the first time; a notebook that repeated that mistake would not be worth
publishing. The order that was actually drawn is recorded in the report.

~20–35 min on a ≤3B model. Most of it is cooldown and the model download.
""")

code(r'''
# ── Run both arms in randomized order ──

import random
import time

import pynvml
import torch

pynvml.nvmlInit()
_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
_gpu_name = pynvml.nvmlDeviceGetName(_handle)
if isinstance(_gpu_name, bytes):
    _gpu_name = _gpu_name.decode()
_gpu_arch = detect_arch(_gpu_name)

print(f"GPU: {_gpu_name} (arch: {_gpu_arch or 'unknown'})")
print(f"Model: {MODEL_NAME} ({PARAMS_B}B) + {PRECISION}")
print("Measuring idle reference (15 s, do not run anything else)...")
IDLE_WATTS, IDLE_TEMP_C = measure_idle_reference(_handle)
pynvml.nvmlShutdown()
print(f"Idle: {IDLE_WATTS:.1f} W, {IDLE_TEMP_C if IDLE_TEMP_C is not None else '?'} C")
print("=" * 62)

ARMS = [PRECISION]
if RUN_FP16_BASELINE and PRECISION != "FP16":
    ARMS.append("FP16")
random.Random().shuffle(ARMS)   # unseeded: the order differs per contributor and is recorded
print(f"Randomized arm order: {' then '.join(ARMS)}")

results = {}
for _i, _arm in enumerate(ARMS, 1):
    print(f"\n[{_i}/{len(ARMS)}] {_arm}")
    _t = time.time()
    try:
        results[_arm] = measure_once(MODEL_NAME, _arm, IDLE_WATTS, IDLE_TEMP_C)
        _r = results[_arm]
        print(
            f"  done in {time.time() - _t:.0f}s — {_r['energy_per_token_mj']:.3f} mJ/token, "
            f"{_r['avg_power_watts']:.1f} W avg, {_r['throughput_tokens_per_s']:.1f} tok/s, "
            f"sampling {_r['sample_rate_hz_achieved']:.0f} Hz achieved"
        )
    except Exception as exc:
        print(f"  {_arm} failed: {exc}")

quant_result = results.get(PRECISION)
fp16_result = results.get("FP16") if PRECISION != "FP16" else None
if quant_result is None:
    raise RuntimeError(f"The {PRECISION} arm did not produce a measurement; nothing to report.")

vs_fp16 = None
if fp16_result:
    base = fp16_result["energy_per_token_mj"]
    vs_fp16 = round((quant_result["energy_per_token_mj"] - base) / base * 100.0, 1)
elif PRECISION != "FP16":
    print("\nNo FP16 baseline -> vs_fp16_energy_pct will be null. The energy/token figure still stands.")

print("\n" + "=" * 62)
print("MEASUREMENT COMPLETE")
print("=" * 62)
''')

md(r"""
## Step 5 · Results and submission

Writes `ecocompute-out/energy.json` and prints text you can paste into an issue on
[ecocompute-mlcube](https://github.com/hongping-zh/ecocompute-mlcube/issues/new).

Submissions are read by hand before anything is added to the published dataset, and some are
not added at all — a run whose cooldown did not converge, or whose achieved sampling rate is
far below the request, tells us more as a bug report than as a data point. Posting one does
not put it on the site automatically.
""")

code(r'''
# ── Display, write energy.json, print submission text ──

import datetime
import json
import os
import platform

_q = quant_result


def _row(label, value):
    print(f"  {label:<16s} {value}")


print("ECOCOMPUTE MEASUREMENT RESULT")
print("-" * 62)
_row("GPU", f"{_gpu_name} ({_gpu_arch or 'unknown'})")
_row("Model", f"{MODEL_NAME} ({PARAMS_B}B)")
_row("Precision", PRECISION)
_row("Energy/token", f"{_q['energy_per_token_mj']:.3f} mJ/token")
_row("Avg power", f"{_q['avg_power_watts']:.1f} W")
_row("Throughput", f"{_q['throughput_tokens_per_s']:.1f} tok/s")
if vs_fp16 is not None:
    _row("vs FP16", f"{vs_fp16:+.1f}%  ({'penalty' if vs_fp16 > 0 else 'saving' if vs_fp16 < 0 else 'neutral'})")
else:
    _row("vs FP16", "n/a (no baseline in this session)")
_row("Arm order", " then ".join(ARMS))
_row("Sampling", f"{_q['sample_rate_hz_achieved']:.0f} Hz achieved of {_q['sample_rate_hz_requested']} requested; "
                 f"reading changed {_q['nvml_update_hz_effective']:.0f} times/s")
print("-" * 62)
print("  GPU package power, not wall AC. n=1. Not a certified benchmark result.")
if not _q["cooldown_before_run"]["converged"]:
    print("  WARNING: the GPU had not returned to idle before this run — treat it as suspect.")

# ── energy.json, in the container's schema ──

report = {
    "schema_version": "ecocompute-energy/1.1",
    "benchmark": "ecocompute-energy-methodology",
    "follows_mlcommons_energy_reporting_conventions": True,
    "certified_benchmark_result": False,
    "scenario": "SingleStream",
    "scenario_note": "Colab notebook measurement, batch=1, not LoadGen-enforced.",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "system_under_test": {
        "gpu": _gpu_name,
        "gpu_arch": _gpu_arch or "unknown",
        "accelerator_count": 1,
        "host": platform.platform(),
        "idle_watts": round(IDLE_WATTS, 1),
        "idle_temp_c": IDLE_TEMP_C,
    },
    "workload": {
        "model_name": MODEL_NAME,
        "params_b": PARAMS_B,
        "params_b_source": "PARAMS_B_OVERRIDE" if PARAMS_B_OVERRIDE is not None else "parsed-from-model-name",
        "precision": PRECISION,
        "batch_size": 1,
        "prompt_tokens": _q["prompt_tokens"],
        "max_new_tokens": 256,
        "context_length": _q["prompt_tokens"] + 256,
    },
    "measurement": {
        "method": "NVML on-device power sampling, trapezoidal integration (Colab notebook)",
        "sample_rate_hz": _q["sample_rate_hz_requested"],
        "sample_rate_hz_note": (
            "sample_rate_hz is the requested rate, kept for schema compatibility. The rate this "
            "run actually achieved is sample_rate_hz_achieved; nvml_update_hz_effective is how "
            "often the NVML reading changed, which bounds the time resolution of the integral."
        ),
        "sample_rate_hz_achieved": _q["sample_rate_hz_achieved"],
        "nvml_update_hz_effective": _q["nvml_update_hz_effective"],
        "tokens_per_run": 256,
        "iterations": 10,
        "warmup": 2,
        "iterations_note": "Decode iterations within one run; this is n=1.",
        "arm_order": list(ARMS),
        "arm_order_note": "Randomized per session; both arms are preceded by a cooldown to idle.",
        "cooldown": {arm: r["cooldown_before_run"] for arm, r in results.items()},
        "integration_note": (
            "Energy is the trapezoidal integral of sampled power. On RTX 4090 the GPU hardware "
            "energy counter read on average 14.9% higher (2.5-31.8%) than integrating the same "
            "traces (doi:10.5281/zenodo.22295184); no counter cross-check is available here."
        ),
    },
    "software": collect_software(),
    "measurement_source": "direct-nvml",
    "results": {
        "total_energy_joules": _q["total_energy_joules"],
        "tokens_generated": _q["tokens_generated"],
        "energy_per_token_mj": _q["energy_per_token_mj"],
        "avg_power_watts": _q["avg_power_watts"],
        "throughput_tokens_per_s": _q["throughput_tokens_per_s"],
        "basis": "measured",
    },
    "provenance": {
        "tool": "https://quantenergy.tech",
        "notebook": "EcoCompute Colab (no-Docker measurement path)",
        "code": "https://github.com/hongping-zh/ecocompute-mlcube",
        "dataset_doi": "10.5281/zenodo.19647290",
    },
    "notice": (
        "Not a certified benchmark result and not an MLPerf result. GPU package power only "
        "(not wall AC). Single observation unless the repeat cell was used."
    ),
}

if fp16_result:
    report["results"]["fp16_energy_per_token_mj"] = fp16_result["energy_per_token_mj"]
    report["results"]["fp16_avg_power_watts"] = fp16_result["avg_power_watts"]
    report["results"]["fp16_throughput_tokens_per_s"] = fp16_result["throughput_tokens_per_s"]
    report["results"]["vs_fp16_energy_pct"] = vs_fp16
else:
    report["results"]["vs_fp16_energy_pct"] = None

os.makedirs("ecocompute-out", exist_ok=True)
with open("ecocompute-out/energy.json", "w") as fh:
    json.dump(report, fh, indent=2)
print("\nSaved: ecocompute-out/energy.json")

# ── Overlay link: the point is encoded in the URL, nothing is uploaded ──

if vs_fp16 is not None:
    import base64

    _payload = {
        "N": PARAMS_B,
        "e": vs_fp16,
        "a": _gpu_arch or "unknown",
        "p": PRECISION,
        "b": "measured",
        "m": MODEL_NAME,
        "g": _gpu_name,
        "s": "Colab notebook, n=1",
    }
    _b64 = base64.urlsafe_b64encode(json.dumps(_payload, separators=(",", ":")).encode()).decode().rstrip("=")
    print(f"Overlay your point on the curve: https://quantenergy.tech/?tab=run&overlay={_b64}")

# ── Submission text ──

vs_line = f"| vs FP16 | {vs_fp16:+.1f}% |" if vs_fp16 is not None else "| vs FP16 | n/a (no baseline) |"
cool_ok = all(r["cooldown_before_run"]["converged"] for r in results.values())

submission = f"""## EcoCompute measurement (community contribution, Colab notebook)

**GPU**: {_gpu_name} ({_gpu_arch or 'unknown'})
**Model**: {MODEL_NAME} ({PARAMS_B}B)
**Precision**: {PRECISION}

### Results
| Metric | Value |
|--------|-------|
| Energy/token | {_q['energy_per_token_mj']:.3f} mJ/token |
| Avg power | {_q['avg_power_watts']:.1f} W |
| Throughput | {_q['throughput_tokens_per_s']:.1f} tok/s |
{vs_line}

### Method
- NVML GPU-package power sampling, trapezoidal integration
- {_q['sample_rate_hz_achieved']:.0f} Hz achieved of {_q['sample_rate_hz_requested']} Hz requested; NVML reading changed {_q['nvml_update_hz_effective']:.0f} times/s
- Batch 1, 256 generated tokens (forced via min_new_tokens), 2 warmup + 10 decode iterations
- Arm order this session: {' then '.join(ARMS)} (randomized), cooldown to idle before each arm: {'converged' if cool_ok else 'DID NOT converge'}
- n=1 unless the repeat cell was used

### Software
- Python {report['software']['python']}
- torch {report['software']['packages'].get('torch', '?')} CUDA {report['software'].get('torch_cuda', '?')}
- bitsandbytes {report['software']['packages'].get('bitsandbytes', '?')}
- NVIDIA driver {report['software'].get('nvidia_driver', '?')}

### Caveats
- GPU-package telemetry, **not** wall AC / whole-system power.
- Energy is integrated from sampled power; on RTX 4090 the hardware energy counter read
  ~14.9% higher than the integral of the same traces. No counter cross-check here.
- Single observation. Not a certified benchmark result, not an MLPerf result.

<details>
<summary>energy.json</summary>

```json
{json.dumps(report, indent=2)}
```
</details>
"""

print("\n" + "=" * 62)
print("Copy the text below into an issue (it is reviewed by hand, not auto-published):")
print("  https://github.com/hongping-zh/ecocompute-mlcube/issues/new")
print("=" * 62)
print(submission)
''')

md(r"""
## Optional · Repeat 5× for a repeatability figure

n=1 has no error bar. This cell repeats **both arms** five times — the order redrawn each
time, cooldown before every arm — and reports the CV of each arm and the spread of
`vs_fp16_energy_pct` across the five paired comparisons. Repeating only the quantized arm
would tell you nothing about the comparison, which is the number people quote.

**Time**: roughly 5× Step 4, so ~2–3 hours on a ≤3B model. Colab will disconnect a free
runtime that it thinks is idle — leave the tab open. Results are written to
`ecocompute-out/energy-repeats.json` as they come in.
""")

code(r'''
# ── Optional: n=5 paired repeats ──

import json
import os
import random
import statistics

N_REPEATS = 5
rng = random.Random()

arms = [PRECISION] + (["FP16"] if RUN_FP16_BASELINE and PRECISION != "FP16" else [])
repeats = []

for rep in range(1, N_REPEATS + 1):
    order = list(arms)
    rng.shuffle(order)
    print(f"\n--- repeat {rep}/{N_REPEATS} — order: {' then '.join(order)} ---")
    rec = {"repeat": rep, "arm_order": list(order), "arms": {}}
    for arm in order:
        try:
            r = measure_once(MODEL_NAME, arm, IDLE_WATTS, IDLE_TEMP_C)
            rec["arms"][arm] = r
            print(f"  {arm}: {r['energy_per_token_mj']:.3f} mJ/token | "
                  f"{r['avg_power_watts']:.1f} W | {r['throughput_tokens_per_s']:.1f} tok/s")
        except Exception as exc:
            print(f"  {arm} failed: {exc}")
    if PRECISION in rec["arms"] and "FP16" in rec["arms"]:
        b = rec["arms"]["FP16"]["energy_per_token_mj"]
        rec["vs_fp16_energy_pct"] = round(
            (rec["arms"][PRECISION]["energy_per_token_mj"] - b) / b * 100.0, 1
        )
        print(f"  vs FP16 this repeat: {rec['vs_fp16_energy_pct']:+.1f}%")
    repeats.append(rec)
    os.makedirs("ecocompute-out", exist_ok=True)
    with open("ecocompute-out/energy-repeats.json", "w") as fh:
        json.dump({"model": MODEL_NAME, "params_b": PARAMS_B, "precision": PRECISION,
                   "gpu": _gpu_name, "repeats": repeats}, fh, indent=2)


def _summarize(label, values):
    if len(values) < 2:
        print(f"{label}: only {len(values)} value(s), no statistics")
        return
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    cv = sd / mean * 100 if mean else float("inf")
    print(f"{label}: {mean:.3f} +/- {sd:.3f} (CV {cv:.2f}%)  n={len(values)}  {[round(v, 3) for v in values]}")


print("\n" + "=" * 62)
for arm in arms:
    _summarize(f"{arm} energy/token (mJ)",
               [r["arms"][arm]["energy_per_token_mj"] for r in repeats if arm in r["arms"]])

deltas = [r["vs_fp16_energy_pct"] for r in repeats if "vs_fp16_energy_pct" in r]
if len(deltas) >= 2:
    mean_d, sd_d = statistics.mean(deltas), statistics.stdev(deltas)
    print(f"vs_fp16_energy_pct: {mean_d:+.1f}% +/- {sd_d:.1f} pp over {len(deltas)} paired repeats  {deltas}")
    print(
        "\nThis spread is run-to-run scatter under this protocol. It is not a confidence\n"
        "interval on the effect: five repeats on one GPU in one session cannot separate\n"
        "the effect from anything that drifted over the session."
    )
print("Saved: ecocompute-out/energy-repeats.json")
''')

nb = {
    "cells": CELLS,
    "metadata": {
        "colab": {"provenance": [], "name": "EcoCompute - Measure LLM Inference Energy", "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU",
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "ecocompute-measure.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out}")
