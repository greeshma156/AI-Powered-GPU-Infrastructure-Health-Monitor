import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulator'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_detector'))


# ── Simulator tests ───────────────────────────────────────────────────────────

class TestGPUSimulator:

    def setup_method(self):
        """Import simulator fresh for each test to avoid Prometheus registry conflicts."""
        import importlib
        import simulator as sim_module
        importlib.reload(sim_module)
        self.GPUSimulator = sim_module.GPUSimulator

    def test_init_correct_number_of_gpus(self):
        sim = self.GPUSimulator(num_gpus=4)
        assert len(sim.gpus) == 4

    def test_gpu_has_required_fields(self):
        sim = self.GPUSimulator(num_gpus=1)
        gpu = sim.gpus[0]
        assert "id" in gpu
        assert "name" in gpu
        assert "mem_total_mb" in gpu
        assert "fault_mode" in gpu

    def test_gpu_name_is_valid_model(self):
        sim = self.GPUSimulator(num_gpus=4)
        valid_names = {"NVIDIA A100-SXM4-80GB", "NVIDIA H100-SXM5-80GB"}
        for gpu in sim.gpus:
            assert gpu["name"] in valid_names

    def test_compute_metrics_returns_expected_keys(self):
        sim = self.GPUSimulator(num_gpus=1)
        metrics = sim._compute_metrics(sim.gpus[0], tick=0)
        expected_keys = ["temp", "util", "mem_util", "mem_used", "power", "sm_clock", "health"]
        for key in expected_keys:
            assert key in metrics

    def test_temperature_within_bounds(self):
        sim = self.GPUSimulator(num_gpus=4)
        for _ in range(100):
            for gpu in sim.gpus:
                m = sim._compute_metrics(gpu, tick=0)
                assert 30 <= m["temp"] <= 95

    def test_utilization_within_bounds(self):
        sim = self.GPUSimulator(num_gpus=4)
        for gpu in sim.gpus:
            m = sim._compute_metrics(gpu, tick=0)
            assert 0 <= m["util"] <= 100

    def test_health_score_within_bounds(self):
        sim = self.GPUSimulator(num_gpus=4)
        for gpu in sim.gpus:
            m = sim._compute_metrics(gpu, tick=0)
            assert 0 <= m["health"] <= 100

    def test_fault_mode_raises_temperature(self):
        sim = self.GPUSimulator(num_gpus=1)
        gpu = sim.gpus[0]
        gpu["fault_mode"] = False
        normal_temps = [sim._compute_metrics(gpu, tick=i)["temp"] for i in range(20)]

        gpu["fault_mode"] = True
        fault_temps = [sim._compute_metrics(gpu, tick=i)["temp"] for i in range(20)]

        assert sum(fault_temps) > sum(normal_temps)

    def test_fault_mode_reduces_health_score(self):
        sim = self.GPUSimulator(num_gpus=1)
        gpu = sim.gpus[0]
        gpu["fault_mode"] = False
        normal_health = [sim._compute_metrics(gpu, tick=i)["health"] for i in range(20)]

        gpu["fault_mode"] = True
        fault_health = [sim._compute_metrics(gpu, tick=i)["health"] for i in range(20)]

        assert sum(fault_health) < sum(normal_health)

    def test_mem_total_is_80gb(self):
        sim = self.GPUSimulator(num_gpus=1)
        assert sim.gpus[0]["mem_total_mb"] == 81920


# ── Detector tests ────────────────────────────────────────────────────────────

class TestDetector:

    def setup_method(self):
        os.environ["GEMINI_API_KEY"] = "test_key"
        os.environ["PROMETHEUS_URL"] = "http://localhost:9090"

    def test_build_prompt_contains_gpu_data(self):
        from detector import build_prompt
        gpus = {
            "0": {"gpu_id": "0", "gpu_name": "NVIDIA A100", "temperature": 55.0}
        }
        prompt = build_prompt(gpus)
        assert "gpu_id" in prompt
        assert "NVIDIA A100" in prompt
        assert "55.0" in prompt

    def test_build_prompt_contains_thresholds(self):
        from detector import build_prompt
        gpus = {"0": {"gpu_id": "0", "gpu_name": "NVIDIA A100"}}
        prompt = build_prompt(gpus)
        assert "75" in prompt     # temp warning threshold
        assert "anomaly_score" in prompt
        assert "JSON" in prompt

    def test_build_prompt_requests_json_output(self):
        from detector import build_prompt
        gpus = {"0": {"gpu_id": "0", "gpu_name": "NVIDIA A100"}}
        prompt = build_prompt(gpus)
        assert "JSON" in prompt
        assert "status" in prompt
        assert "findings" in prompt

    def test_call_gemini_returns_empty_without_key(self):
        os.environ["GEMINI_API_KEY"] = ""
        from detector import call_gemini_api
        import importlib
        import detector
        importlib.reload(detector)
        result = detector.call_gemini_api("test prompt")
        assert result == {}

    def test_collect_snapshot_returns_dict(self):
        """collect_gpu_snapshot should return a dict (empty if Prometheus unavailable)."""
        from detector import collect_gpu_snapshot
        result = collect_gpu_snapshot()
        assert isinstance(result, dict)
