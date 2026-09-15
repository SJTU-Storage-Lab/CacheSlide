"""Repository/import boundaries prevent vendored-engine shadowing and CUDA work."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_legacy_engine_or_build_system_is_shipped():
    for name in ("vllm", "csrc", "cmake", "setup.py", "CMakeLists.txt"):
        assert not (ROOT / name).exists(), name
    assert (ROOT / "src/cacheslide_vllm/compatibility.json").is_file()
    assert (ROOT / "run_cacheslide_benchmark.sh").is_file()


def test_configuration_and_planning_import_without_numeric_or_native_engines():
    code = r"""
import importlib.abc
import sys
class BlockEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'safetensors', 'vllm'}:
            raise AssertionError('unexpected engine import: ' + fullname)
sys.meta_path.insert(0, BlockEngines())
sys.path.insert(0, sys.argv[1])
from cacheslide_vllm import cli, config, contracts, policy, workflow
assert policy.WCAConfig().correction_fraction == 0.26
config.CacheSlideSettings('/adapter', '/cache')
assert cli.parser() and workflow.parser()
assert not {'torch', 'safetensors', 'vllm'} & set(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(ROOT / "src")],
        check=True,
        timeout=20,
    )
