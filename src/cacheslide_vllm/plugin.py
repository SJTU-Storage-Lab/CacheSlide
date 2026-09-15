"""vLLM general-plugin entry point; registration does not enable CacheSlide."""


def register() -> None:
    from vllm import ModelRegistry

    architecture = "CacheSlideLlamaForCausalLM"
    if architecture not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            architecture, "cacheslide_vllm.model:CacheSlideLlamaForCausalLM"
        )
