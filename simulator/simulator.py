import os
import time
import random
import math
import logging
from prometheus_client import start_http_server, Gauge, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Prometheus metrics (DCGM-style naming) ──────────────────────────────────
GPU_TEMP          = Gauge("dcgm_gpu_temp_celsius",       "GPU temperature in Celsius",        ["gpu_id", "gpu_name"])
GPU_UTILIZATION   = Gauge("dcgm_gpu_utilization_percent","GPU compute utilization (%)",        ["gpu_id", "gpu_name"])
MEM_UTILIZATION   = Gauge("dcgm_mem_utilization_percent","GPU memory utilization (%)",         ["gpu_id", "gpu_name"])
MEM_USED_MB       = Gauge("dcgm_mem_used_mb",            "GPU memory used (MB)",               ["gpu_id", "gpu_name"])
MEM_TOTAL_MB      = Gauge("dcgm_mem_total_mb",           "GPU memory total (MB)",              ["gpu_id", "gpu_name"])
POWER_USAGE_W     = Gauge("dcgm_power_usage_watts",      "GPU power draw (W)",                 ["gpu_id", "gpu_name"])
SM_CLOCK_MHZ      = Gauge("dcgm_sm_clock_mhz",          "GPU SM clock speed (MHz)",           ["gpu_id", "gpu_name"])
HEALTH_SCORE      = Gauge("dcgm_gpu_health_score",       "AI-computed GPU health score 0-100", ["gpu_id", "gpu_name"])
XID_ERRORS        = Counter("dcgm_xid_errors_total",     "Total Xid (GPU error) events",       ["gpu_id", "gpu_name", "error_type"])
PCIE_ERRORS       = Counter("dcgm_pcie_replay_errors_total", "PCIe replay error count",        ["gpu_id", "gpu_name"])


class GPUSimulator:
    """
    Simulates DCGM-style telemetry for N GPUs.
    Each GPU has a baseline state plus random drift and occasional fault injection.
    """

    GPU_MODELS = ["NVIDIA A100-SXM4-80GB", "NVIDIA H100-SXM5-80GB"]

    def __init__(self, num_gpus: int = 4):
        self.num_gpus = num_gpus
        self.gpus = [self._init_gpu(i) for i in range(num_gpus)]
        logger.info(f"Initialised {num_gpus} simulated GPUs")

    def _init_gpu(self, gpu_id: int) -> dict:
        return {
            "id": str(gpu_id),
            "name": random.choice(self.GPU_MODELS),
            "mem_total_mb": 81920,          # 80 GB
            "base_temp": random.uniform(40, 55),
            "base_util": random.uniform(20, 60),
            "fault_mode": False,            # True = injecting pre-failure pattern
            "fault_timer": 0,
        }

    def _maybe_inject_fault(self, gpu: dict):
        """Randomly flip a GPU into fault mode to simulate pre-failure behaviour."""
        if not gpu["fault_mode"] and random.random() < 0.002:   # ~0.2% chance per tick
            gpu["fault_mode"] = True
            gpu["fault_timer"] = random.randint(10, 30)          # lasts 10–30 ticks
            logger.warning(f"[GPU {gpu['id']}] Fault injected — simulating pre-failure pattern")
        elif gpu["fault_mode"]:
            gpu["fault_timer"] -= 1
            if gpu["fault_timer"] <= 0:
                gpu["fault_mode"] = False
                logger.info(f"[GPU {gpu['id']}] Fault cleared")

    def _compute_metrics(self, gpu: dict, tick: int) -> dict:
        fault = gpu["fault_mode"]
        wave  = math.sin(tick * 0.1)               # gentle sinusoidal drift

        temp = gpu["base_temp"] + wave * 5 + random.uniform(-2, 2)
        util = gpu["base_util"] + wave * 10 + random.uniform(-5, 5)
        mem_util = util * 0.85 + random.uniform(-3, 3)

        if fault:
            temp     += random.uniform(15, 30)     # thermal spike
            util     += random.uniform(10, 25)     # utilisation spike
            mem_util += random.uniform(10, 20)     # memory pressure

        temp     = max(30, min(temp, 95))
        util     = max(0,  min(util, 100))
        mem_util = max(0,  min(mem_util, 100))

        power_w     = 150 + util * 2.5 + (30 if fault else 0) + random.uniform(-10, 10)
        sm_clock    = 1410 - (30 if fault else 0) + random.randint(-50, 50)
        mem_used_mb = gpu["mem_total_mb"] * mem_util / 100

        # health score: 100 = healthy, degrades with temp / fault
        health = 100
        if temp > 80:
            health -= (temp - 80) * 3
        if fault:
            health -= random.uniform(20, 45)
        health = max(0, min(health, 100))

        return {
            "temp":      temp,
            "util":      util,
            "mem_util":  mem_util,
            "mem_used":  mem_used_mb,
            "power":     power_w,
            "sm_clock":  sm_clock,
            "health":    health,
            "xid_error": fault and random.random() < 0.3,
            "pcie_error": random.random() < 0.01,
        }

    def tick(self, tick: int):
        for gpu in self.gpus:
            self._maybe_inject_fault(gpu)
            m = self._compute_metrics(gpu, tick)
            labels = [gpu["id"], gpu["name"]]

            GPU_TEMP.labels(*labels).set(m["temp"])
            GPU_UTILIZATION.labels(*labels).set(m["util"])
            MEM_UTILIZATION.labels(*labels).set(m["mem_util"])
            MEM_USED_MB.labels(*labels).set(m["mem_used"])
            MEM_TOTAL_MB.labels(*labels).set(gpu["mem_total_mb"])
            POWER_USAGE_W.labels(*labels).set(m["power"])
            SM_CLOCK_MHZ.labels(*labels).set(m["sm_clock"])
            HEALTH_SCORE.labels(*labels).set(m["health"])

            if m["xid_error"]:
                XID_ERRORS.labels(*labels, error_type="DBE").inc()
                logger.warning(f"[GPU {gpu['id']}] Xid DBE error fired")

            if m["pcie_error"]:
                PCIE_ERRORS.labels(*labels).inc()


def main():
    num_gpus        = int(os.getenv("NUM_GPUS", 4))
    scrape_interval = int(os.getenv("SCRAPE_INTERVAL", 15))
    port            = int(os.getenv("METRICS_PORT", 8000))

    start_http_server(port)
    logger.info(f"Prometheus metrics server started on :{port}")

    sim  = GPUSimulator(num_gpus=num_gpus)
    tick = 0

    while True:
        sim.tick(tick)
        tick += 1
        time.sleep(scrape_interval)


if __name__ == "__main__":
    main()
