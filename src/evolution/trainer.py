"""Unsloth LoRA fine-tuning wrapper for Phase 1 (single-lineage evolution)."""

import unsloth  # noqa: F401 — must be first import for optimizations
from unsloth import FastLanguageModel

import json
import logging
import os
from dataclasses import asdict

import torch

from src.models.adapters import AdapterMetadata
from src.metrics.core import (
    disorganization_entropy,
    effective_complexity,
    fitness_score,
    mutual_information_proxy,
    shannon_entropy,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase-1 constants (frozen)
# ---------------------------------------------------------------------------

_BASE_MODEL: str = "unsloth/qwen3-8b-base-unsloth-bnb-4bit"
_LORA_R: int = 8
_LORA_ALPHA: int = 16
_LORA_TARGET_MODULES: list[str] = ["q_proj", "v_proj"]
_MAX_SEQ_LENGTH: int = 512
_MAX_NEW_TOKENS: int = 200
_BASE_MODEL_CACHE: dict = {"model": None, "tokenizer": None}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TrainingError(RuntimeError):
    """Raised when Unsloth SFTTrainer fails during adapter training.

    Parameters
    ----------
    agent_id : str
        Identifier of the agent whose training failed.
    epoch : int
        Training epoch at which the failure occurred (0-indexed).
    cause : BaseException
        The underlying exception.
    """

    def __init__(self, agent_id: str, epoch: int, cause: BaseException) -> None:
        super().__init__(
            f"Training failed for agent_id={agent_id!r} at epoch {epoch}: {cause}"
        )
        self.agent_id = agent_id
        self.epoch = epoch
        self.cause = cause


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _log_gpu_memory(tag: str) -> None:
    """Log current CUDA memory allocation at INFO level."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        log.info(
            "%s — GPU memory: %.1f MB allocated, %.1f MB reserved",
            tag,
            allocated,
            reserved,
        )
    else:
        log.info("%s — CUDA not available, running on CPU", tag)


def _resolve_base_model_name(config: dict) -> str:
    """Return the Unsloth 4-bit model name, ignoring Ollama model names."""
    name: str = config.get("model", {}).get("base_model", _BASE_MODEL)
    # Phase-1 config stores the Ollama model name; always use the Unsloth
    # 4-bit variant for local fine-tuning.
    if not name.startswith("unsloth/"):
        return _BASE_MODEL
    return name


def get_base_model(base_model_name: str) -> tuple:
    """Load the base model once and return the cached model/tokenizer pair."""
    global _BASE_MODEL_CACHE

    if _BASE_MODEL_CACHE["model"] is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=_MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        _BASE_MODEL_CACHE["model"] = model
        _BASE_MODEL_CACHE["tokenizer"] = tokenizer
        log.info("Base model loaded and cached: %s", base_model_name)

    return _BASE_MODEL_CACHE["model"], _BASE_MODEL_CACHE["tokenizer"]


def _build_fresh_lora(model, seed: int):
    """Attach a fresh LoRA adaptor with Phase-1 fixed hyper-parameters.

    Parameters
    ----------
    model :
        Base model returned by ``FastLanguageModel.from_pretrained``.
    seed : int
        RNG seed for LoRA weight initialisation.

    Returns
    -------
    PEFT model ready for training.
    """
    return FastLanguageModel.get_peft_model(
        model,
        r=_LORA_R,
        lora_alpha=_LORA_ALPHA,
        target_modules=_LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_adapter(
    documents: list[str],
    parent_adapter_path: str | None,
    output_dir: str,
    agent_id: str,
    metadata: AdapterMetadata,
    config: dict,
) -> str:
    """Fine-tune a LoRA adapter on the supplied documents for one epoch.

    If *parent_adapter_path* is provided the parent adapter is loaded as a
    trainable PEFT model on top of the 4-bit base, so the new generation
    initialises from the parent's learned weights.  If it is ``None`` a fresh
    LoRA is initialised from scratch.

    Documents are tokenised as plain text — no chat template, no labels.

    GPU memory is released for the trainable LoRA wrapper before the function
    returns. The cached base model remains resident for reuse.

    Parameters
    ----------
    documents : list[str]
        Plain-text training documents.
    parent_adapter_path : str or None
        Path to a parent adapter directory, or ``None`` for base model.
    output_dir : str
        Root directory; the adapter is saved to ``output_dir/adapter_{agent_id}/``.
    agent_id : str
        Agent identifier, used for directory naming and log messages.
    metadata : AdapterMetadata
        Metadata persisted alongside the adapter weights as ``metadata.json``.
    config : dict
        Full experiment config dict (reads ``project.seed`` and
        ``model.base_model``).

    Returns
    -------
    str
        Absolute path to the saved adapter directory.

    Raises
    ------
    FileNotFoundError
        If *parent_adapter_path* is given but does not exist on disk.
    TrainingError
        If training fails for any reason.
    """
    from datasets import Dataset  # type: ignore
    from peft import PeftModel  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    seed: int = int(config.get("project", {}).get("seed", 42))
    base_model_name: str = _resolve_base_model_name(config)
    save_path: str = os.path.join(output_dir, f"adapter_{agent_id}")
    os.makedirs(save_path, exist_ok=True)

    log.info(
        "train_adapter START — agent_id=%s, documents=%d, parent=%s",
        agent_id,
        len(documents),
        parent_adapter_path or "none (fresh base)",
    )
    _log_gpu_memory("before-training")

    peft_model = None
    trainer = None
    try:
        # ------------------------------------------------------------------
        # 1. Load 4-bit base model
        # ------------------------------------------------------------------
        base_model, tokenizer = get_base_model(base_model_name)

        # ------------------------------------------------------------------
        # 2. Attach LoRA adaptor
        #    • parent provided → load existing adapter as trainable so the
        #      new generation initialises from parent weights (continued
        #      training).  Gradient checkpointing is re-enabled explicitly.
        #    • no parent → fresh LoRA via Unsloth helper (sets up gradient
        #      checkpointing internally).
        # ------------------------------------------------------------------
        if parent_adapter_path is not None:
            if not os.path.isdir(parent_adapter_path):
                raise FileNotFoundError(
                    f"Parent adapter directory not found: {parent_adapter_path!r}"
                )
            log.info("Loading parent adapter from %s", parent_adapter_path)
            peft_model = PeftModel.from_pretrained(
                base_model, parent_adapter_path, is_trainable=True
            )
            # Unsloth disables input-gradient hooks on the base model by
            # default; re-enable them so LoRA receives gradients.
            peft_model.enable_input_require_grads()
            peft_model.gradient_checkpointing_enable()
        else:
            peft_model = _build_fresh_lora(base_model, seed)

        # ------------------------------------------------------------------
        # 3. Build HuggingFace Dataset from plain documents
        # ------------------------------------------------------------------
        dataset = Dataset.from_dict({"text": documents})

        # ------------------------------------------------------------------
        # 4. Train for exactly 1 epoch
        # ------------------------------------------------------------------
        tmp_output = os.path.join(save_path, "trainer_tmp")
        training_args = SFTConfig(
            output_dir=tmp_output,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            seed=seed,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            dataset_text_field="text",
            max_seq_length=_MAX_SEQ_LENGTH,
            packing=True,
        )

        trainer = SFTTrainer(
            model=peft_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

        trainer.train()

    except FileNotFoundError:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        raise TrainingError(agent_id=agent_id, epoch=0, cause=exc) from exc

    # ------------------------------------------------------------------
    # 5. Save adapter weights and metadata
    # ------------------------------------------------------------------
    peft_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    metadata_path = os.path.join(save_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(metadata), fh, indent=2)

    log.info("Adapter saved to %s", save_path)
    _log_gpu_memory("after-training")

    # ------------------------------------------------------------------
    # 6. Detach the trainable wrapper and keep the pristine base in cache
    # ------------------------------------------------------------------
    if peft_model is not None and hasattr(peft_model, "unload"):
        _BASE_MODEL_CACHE["model"] = peft_model.unload()

    del trainer
    del peft_model
    torch.cuda.empty_cache()
    log.info("train_adapter END — agent_id=%s", agent_id)

    return os.path.abspath(save_path)


def load_adapter(
    adapter_path: str,
    base_model_name: str,
) -> tuple:
    """Load a saved LoRA adapter on top of the 4-bit base model.

    Parameters
    ----------
    adapter_path : str
        Path to the adapter directory produced by :func:`train_adapter`.
    base_model_name : str
        HuggingFace or Ollama model name.  Non-Unsloth names are resolved to
        the project default (``unsloth/qwen3-8b-base-unsloth-bnb-4bit``).

    Returns
    -------
    tuple
        ``(model, tokenizer)`` ready for inference.

    Raises
    ------
    FileNotFoundError
        If *adapter_path* does not exist on disk.
    """
    from peft import PeftModel  # type: ignore

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path!r}. "
            "Ensure train_adapter() completed successfully before calling load_adapter()."
        )

    if not base_model_name.startswith("unsloth/"):
        base_model_name = _BASE_MODEL

    log.info("load_adapter — loading base model %s", base_model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=_MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    log.info("load_adapter — applying adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    FastLanguageModel.for_inference(model)

    log.info("load_adapter — model ready for inference")
    return model, tokenizer


def measure_metrics(
    model,
    tokenizer,
    diagnostic_prompt: str,
    seed_output: str,
    fitness_weights: dict,
) -> dict:
    """Generate model output and compute Phase 0/1 information-theoretic metrics.

    Generation uses greedy decoding (``do_sample=False``,
    ``max_new_tokens=200``).  Only newly generated tokens are decoded and
    evaluated; the prompt tokens are excluded.

    Parameters
    ----------
    model :
        Loaded model returned by :func:`load_adapter`.
    tokenizer :
        Matching tokenizer returned by :func:`load_adapter`.
    diagnostic_prompt : str
        Input prompt fed to the model.
    seed_output : str
        Reference output used as the seed for the I(X;seed) proxy.
    fitness_weights : dict
        Must contain keys ``"w1"``, ``"w2"``, ``"w3"`` (floats).
        Frozen Phase-1 values: w1=0.3, w2=0.5, w3=0.2.

    Returns
    -------
    dict
        Keys: ``h_x``, ``c_x``, ``i_x_seed``, ``h_dezorg``, ``fitness``.
    """
    w1: float = float(fitness_weights["w1"])
    w2: float = float(fitness_weights["w2"])
    w3: float = float(fitness_weights["w3"])

    inputs = tokenizer(diagnostic_prompt, return_tensors="pt").to(model.device)
    prompt_len: int = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
        )

    # Decode only the newly generated tokens (exclude prompt)
    new_token_ids = output_ids[0][prompt_len:]
    output_text: str = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    h_x = shannon_entropy(output_text)
    c_x = effective_complexity(output_text)
    i_x_seed = mutual_information_proxy(seed_output, output_text)
    h_dezorg = disorganization_entropy(output_text)
    f = fitness_score(c_x, i_x_seed, h_dezorg, w1, w2, w3)

    return {
        "h_x": h_x,
        "c_x": c_x,
        "i_x_seed": i_x_seed,
        "h_dezorg": h_dezorg,
        "fitness": f,
    }
