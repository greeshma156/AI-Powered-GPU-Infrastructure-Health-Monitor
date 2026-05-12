import os
import time
import logging
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMETHEUS_URL    = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
ANOMALY_THRESHOLD = int(os.getenv("ANOMALY_THRESHOLD", 80))   # score above this triggers recovery
CHECK_INTERVAL    = int(os.getenv("CHECK_INTERVAL", 30))       # seconds between checks

# Track which GPUs have already had recovery triggered (avoid repeat actions)
recovered_gpus = {}


def fetch_anomaly_scores() -> list:
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "ai_anomaly_score"},
            timeout=5
        )
        response.raise_for_status()
        return response.json().get("data", {}).get("result", [])
    except Exception as e:
        logger.error(f"Failed to fetch anomaly scores: {e}")
        return []


def simulate_recovery(gpu_id: str, gpu_name: str, score: float):
    """
    In a real system this would:
    - Reset the GPU driver
    - Drain workloads off the GPU
    - Page the on-call engineer
    - Submit a hardware ticket

    Here we simulate those actions with log output.
    """
    logger.warning(f"[RECOVERY TRIGGERED] GPU {gpu_id} ({gpu_name}) — anomaly score {score}")
    logger.info(f"  Step 1: Draining workloads from GPU {gpu_id}...")
    time.sleep(1)
    logger.info(f"  Step 2: Resetting GPU {gpu_id} driver state...")
    time.sleep(1)
    logger.info(f"  Step 3: Flagging GPU {gpu_id} as quarantined in fleet registry...")
    time.sleep(1)
    logger.info("  Step 4: Alerting on-call engineer via Alertmanager...")
    time.sleep(1)
    logger.info(f"  Step 5: Submitting hardware inspection ticket for GPU {gpu_id}...")
    logger.info(f"[RECOVERY COMPLETE] GPU {gpu_id} quarantined at {datetime.utcnow().isoformat()}")


def run_recovery_loop():
    logger.info(f"Auto-recovery agent started. Threshold={ANOMALY_THRESHOLD}, Interval={CHECK_INTERVAL}s")

    while True:
        scores = fetch_anomaly_scores()

        for result in scores:
            gpu_id   = result["metric"].get("gpu_id", "unknown")
            gpu_name = result["metric"].get("gpu_name", "unknown")
            score    = float(result["value"][1])

            if score >= ANOMALY_THRESHOLD:
                last_recovery = recovered_gpus.get(gpu_id, 0)
                # Only trigger recovery once every 5 minutes per GPU
                if time.time() - last_recovery > 300:
                    simulate_recovery(gpu_id, gpu_name, score)
                    recovered_gpus[gpu_id] = time.time()
                else:
                    logger.info(f"GPU {gpu_id} still in recovery cooldown — skipping")
            else:
                # Clear recovery state if GPU recovers
                if gpu_id in recovered_gpus:
                    logger.info(f"GPU {gpu_id} recovered — anomaly score back to {score}")
                    del recovered_gpus[gpu_id]

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_recovery_loop()
