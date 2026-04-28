from src.models.adapters import adapter_filename, build_agent_id


def test_build_agent_id_format() -> None:
    assert build_agent_id("ego", generation=2, index=7) == "ego_gen02_007"


def test_adapter_filename_format() -> None:
    assert adapter_filename("ego_gen02_007") == "adapter_ego_gen02_007.safetensors"
