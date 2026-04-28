import os
os.environ["HTTPS_PROXY"] = "http://172.17.192.1:8080"
os.environ["HTTP_PROXY"] = "http://172.17.192.1:8080"

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import math

# --- Proxy setup for HuggingFace Hub ---
os.environ["http_proxy"] = "http://172.29.224.1:8080"
os.environ["https_proxy"] = "http://172.29.224.1:8080"

# --- Model selection ---
MODEL_ID = "Qwen/Qwen3-8B-Base"
FALLBACK_MODEL_ID = "Qwen/Qwen3-4B-Base"

# --- Input text ---
PROMPT = """The quick brown fox jumps over the lazy dog. This is a test sentence for log probability extraction. Language models can be evaluated using perplexity."""

def try_load_model(model_id):
    print(f"Loading model: {model_id} (4bit, Unsloth)")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_id,
            load_in_4bit = True,
            device_map = "auto"
        )
        return model, tokenizer, model_id
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return None, None, model_id
# Try 8B, fallback to 4B if OOM
model, tokenizer, used_model_id = try_load_model(MODEL_ID)
if model is None:
    model, tokenizer, used_model_id = try_load_model(FALLBACK_MODEL_ID)
    if model is None:
        print("ERROR: Could not load either 8B or 4B model. Aborting.")
        exit(1)
print(f"\nModel loaded: {used_model_id}")

# --- Tokenize input ---
inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# --- Move to correct device ---
device = next(model.parameters()).device
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
# --- Forward pass ---
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
# --- Compute logprobs ---
# Shift input_ids and logits for next-token prediction
shifted_logits = logits[:, :-1, :]
shifted_labels = input_ids[:, 1:]
log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
# Gather logprobs for the actual next tokens
next_token_logprobs = log_probs.gather(2, shifted_labels.unsqueeze(-1)).squeeze(-1)

# Convert to list for printing
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
logprobs_list = next_token_logprobs[0].tolist()

print("\nToken-level log probabilities:")
for i, (tok, lp) in enumerate(zip(tokens[1:], logprobs_list)):
    print(f"{i:2d} | {tok:15} | {lp: .4f}")

# --- Perplexity calculation ---
perplexity = math.exp(-sum(logprobs_list) / len(logprobs_list))
print(f"\nPerplexity: {perplexity:.4f}")
print(f"(Model used: {used_model_id})")
import requests
import json

# Configuration
OLLAMA_URL = "http://172.29.224.1:11434/api/generate"
MODEL = "qwen3:8b-base"
PROMPT = "The quick brown fox jumps over the lazy dog."

payload = {
    "model": MODEL,
    "prompt": PROMPT,
    "options": {
        "logprobs": True
    },
    "stream": False
}

print(f"Sending request to Ollama at {OLLAMA_URL} with logprobs enabled...")
response = requests.post(OLLAMA_URL, json=payload)

try:
    response.raise_for_status()
    data = response.json()
    print("\nRaw response:")
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Parse and print token-level logprobs if present
logprobs = None
if "logprobs" in data:
    logprobs = data["logprobs"]
    print("\nToken-level logprobs:")
    for i, lp in enumerate(logprobs):
        print(f"Token {i}: {lp}")
    # Example perplexity calculation (H_dezorg proxy)
    if logprobs:
        avg_neg_logprob = -sum(logprobs) / len(logprobs)
        print(f"\nAverage negative logprob (perplexity proxy): {avg_neg_logprob}")
else:
    print("\nNo 'logprobs' field in response. Check the raw response above.")
    print("If logprobs are missing, review Ollama/model support or propose alternative.")
