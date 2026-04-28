import json

from src.analysis.phase0_metric_validation import run_phase0_metric_validation


def test_run_phase0_metric_validation_writes_output(tmp_path) -> None:
    config = {
        "project": {"name": "evolutionary-llm-research", "phase": 0, "seed": 7},
        "constraints": {
            "pass_content_type_tags_to_model": False,
            "phase_mechanics_isolated": True,
        },
        "biomes": {
            "plain": {"food": 0.8, "predator": 0.05, "noise": 0.15, "k_max": 25},
        },
        "mechanics": {
            "reproduction_enabled": False,
            "hgt_enabled": False,
            "cannibalism_enabled": False,
        },
        "metrics": {
            "weights": {
                "w1_complexity": 1.0,
                "w2_mutual_info": 1.0,
                "w3_disorganization": 1.0,
            }
        },
        "phase0_validation": {
            "seed_text": "seed baseline text",
            "samples": [
                {"id": "s1", "text": "example output text one."},
                {"id": "s2", "text": "example output text two."},
            ],
        },
    }

    output_path = run_phase0_metric_validation(config=config, output_root=tmp_path)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["project"]["phase"] == 0
    assert payload["run"]["sample_count"] == 2
    assert len(payload["results"]) == 2
