"""Explicit opt-in native worker; imported only in a vLLM worker process."""

from vllm.v1.worker.gpu_worker import Worker

from .compat import validate_engine_config, verify_installed_vllm
from .integration import attach_runner


class CacheSlideWorker(Worker):
    def cacheslide_metrics(self):
        """Small, request-text-free receipt queried outside benchmark timing."""
        model = self.model_runner.get_model()
        return dict(model.model.cacheslide_runtime.last_metrics)

    def cacheslide_close(self):
        """Drain this opt-in store and remove only its own writer lock."""
        runner = getattr(self, "model_runner", None)
        if runner is not None and getattr(runner, "model", None) is not None:
            model = runner.get_model()
            runtime = getattr(getattr(model, "model", None), "cacheslide_runtime", None)
            if runtime is not None:
                runtime.close()

    def shutdown(self):
        try:
            self.cacheslide_close()
        finally:
            super().shutdown()

    def __init__(
        self,
        vllm_config,
        local_rank,
        rank,
        distributed_init_method,
        is_driver_worker=False,
    ):
        verify_installed_vllm()
        validate_engine_config(vllm_config)
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

    def init_device(self):
        super().init_device()
        attach_runner(self.model_runner, use_v2=self.use_v2_model_runner)
