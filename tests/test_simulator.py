import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulator'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_detector'))


# ── Simulator tests ───────────────────────────────────────────────────────────

class TestGPUSimulator:

    def setup_method(self):
        """Use a fresh Prometheus registry for each test to avoid duplicate metric errors."""
        from prometheus_client import CollectorRegistry
        import simulator
        # Patch the registry before importing metrics
        self.registry = CollectorRegistry()
        simulator.GPU_TEMP._labelnames  # touch to confirm loaded
        self.GPUSimulator = simulator.GPUSimulator

    def _fresh_simulator(self, num_gpus=4):
        """Import simulator fresh with a clean registry each time."""
        import importlib
        from prometheus_client import CollectorRegistry, REGISTRY
        # Unregister all collectors to get a clean state
        collectors = list(REGISTRY._names_to_collectors.values())
        for c in set(collectors):
            try:
                REGISTRY.unregister(c)
            except Exception:
                pass
        import simulator
        importlib.reload(simulator)
        return simulator.GPUSimulator(num_gpus=num_gpus)

    def test_init_correct_number_of_gpus(self):
        sim = self._fresh_simulator(num_gpus=4)
        assert len(sim.gpus) == 4

    def test_gpu_has_required_fields(self):
        sim = self._fresh_simulator(num_gpus=1)
        gpu = sim.gpus[0]
        assert "id" in gpu
        assert "name" in gpu
        assert "mem_total_mb" in gpu
        assert "fault_mode" in gpu

    def test_gpu_name_is_valid_model(self):
        sim = self._fresh_simulator(num_gpus=4)
        valid_names = {"NVIDIA A100-SXM4-80GB", "NVIDIA H100-SXM5-80GB"}
        for gpu in sim.gpus:
            assert gpu["name"] in valid_names

    def test_compute_metrics_returns_expected_keys(self):
        sim = self._fresh_simulator(num_gpus=1)
        metrics = sim._compute_metrics(sim.gpus[0], tick=0)
        for key in ["temp", "util", "mem_util", "mem_used", "power", "sm_clock", "health"]:
            assert key in metrics

    def test_temperature_within_bounds(self):
        sim = self._fresh_simulator(num_gpus=1)
        for i in range(20):
            m = sim._compute_metrics(sim.gpus[0], tick=i)
            assert 30 <= m["temp"] <= 95

    def test_utilization_within_bounds(self):
        sim = self._fresh_simulator(num_gpus=1)
        m = sim._compute_metrics(sim.gpus[0], tick=0)
        assert 0 <= m["util"] <= 100

    def test_health_score_within_bounds(self):
        sim = self._fresh_simulator(num_gpus=1)
        m = sim._compute_metrics(sim.gpus[0], tick=0)
        assert 0 <= m["health"] <= 100

    def test_fault_mode_raises_temperature(self):
        sim = self._fresh_simulator(num_gpus=1)
        gpu = sim.gpus[0]
        gpu["fault_mode"] = False
        normal_temps = [sim._compute_metrics(gpu, tick=i)["temp"] for i in range(20)]
        gpu["fault_mode"] = True
        fault_temps = [sim._compute_metrics(gpu, tick=i)["temp"] for i in range(20)]
        assert sum(fault_temps) > sum(normal_temps)

    def test_fault_mode_reduces_health_score(self):
        sim = self._fresh_simulator(num_gpus=1)
        gpu = sim.gpus[0]
        gpu["fault_mode"] = False
        normal_health = [sim._compute_metrics(gpu, tick=i)["health"] for i in range(20)]
        gpu["fault_mode"] = True
        fault_health = [sim._compute_metrics(gpu, tick=i)["health"] for i in range(20)]
        assert sum(fault_health) < sum(normal_health)

    def test_mem_total_is_80gb(self):
        sim = self._fresh_simulator(num_gpus=1)
        assert sim.gpus[0]["mem_total_mb"] == 81920


# ── Detector tests ────────────────────────────────────────────────────────────

class TestDetector:

    def test_build_prompt_contains_gpu_data(self):
        os.environ["GEMINI_API_KEY"] = "test_key"
        os.environ["PROMETHEUS_URL"] = "http://localhost:9090"
        from detector import build_prompt
        gpus = {"0": {"gpu_id": "0", "gpu_name": "NVIDIA A100", "temperature": 55.0}}
        prompt = build_prompt(gpus)
        assert "gpu_id" in prompt
        assert "NVIDIA A100" in prompt

    def test_build_prompt_contains_thresholds(self):
        from detector import build_prompt
        gpus = {"0": {"gpu_id": "0", "gpu_name": "NVIDIA A100"}}
        prompt = build_prompt(gpus)
        assert "75" in prompt
        assert "anomaly_score" in prompt

    def test_build_prompt_requests_json_output(self):
        from detector import build_prompt
        gpus = {"0": {"gpu_id": "0", "gpu_name": "NVIDIA A100"}}
        prompt = build_prompt(gpus)
        assert "JSON" in prompt
        assert "findings" in prompt

    def test_collect_snapshot_returns_dict(self):
        from detector import collect_gpu_snapshot
        result = collect_gpu_snapshot()
        assert isinstance(result, dict)
