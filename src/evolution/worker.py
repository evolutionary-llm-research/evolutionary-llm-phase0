import argparse
import json
import sys
import os

# Phase 1 frozen constants (must match biome_runner.py)
DIAGNOSTIC_PROMPT: str = (
    "Summarize the key mechanisms by which misinformation spreads "
    "in online environments and describe evidence-based interventions."
)
FITNESS_WEIGHTS: dict[str, float] = {"w1": 0.3, "w2": 0.5, "w3": 0.2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--parent-adapter", default="none")
    parser.add_argument("--docs-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--seed-output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-json", required=True)
    args = parser.parse_args()

    from src.evolution.trainer import train_and_measure
    from src.models.adapters import AdapterMetadata
    import yaml

    with open(args.docs_file) as f:
        documents = json.load(f)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    parent = None if args.parent_adapter == "none" else args.parent_adapter
    
    # Reconstruct AdapterMetadata from JSON
    metadata_dict = json.loads(args.metadata_json)
    metadata = AdapterMetadata(**metadata_dict)

    try:
        adapter_path, metrics = train_and_measure(
            documents=documents,
            parent_adapter_path=parent,
            output_dir=args.output_dir,
            agent_id=args.agent_id,
            metadata=metadata,
            config=config,
            diagnostic_prompt=DIAGNOSTIC_PROMPT,
            seed_output=args.seed_output,
            fitness_weights=FITNESS_WEIGHTS,
            base_model_name=args.base_model,
        )
        result = {"status": "ok", "agent_id": args.agent_id,
                  "adapter_path": adapter_path, "metrics": metrics}
    except Exception as e:
        result = {"status": "error", "agent_id": args.agent_id, "error": str(e)}
        with open(args.output_file, "w") as f:
            json.dump(result, f)
        sys.exit(1)

    with open(args.output_file, "w") as f:
        json.dump(result, f)
    sys.exit(0)

if __name__ == "__main__":
    main()
