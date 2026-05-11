import os
import time
import json
import logging
import requests
from datetime import datetime
from prometheus_client import start_http_server, Gauge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AI_ANOMALY_SCORE = Gauge(
    "ai_anomaly_score",
    "AI-detected anomaly score per GPU (0=normal, 100=critical)",
    ["gpu_id", "gpu_name"]
)
AI_ANALYSIS_TIMESTAMP = Gauge(
    "ai_last_analysis_timestamp",
    "Unix timestamp of last AI analysis",
    ["gpu_id"]
)

PROMETHEUS_URL    = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
ANALYSIS_INTERVAL = int(os.getenv("ANALYSIS_INTERVAL", 60))
METRICS_PORT      = int(os.getenv("METRICS_PORT", 8001))
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"


def fetch_metric(query):
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        response.raise_for_status()
        return response.json().get("data", {}).get("result", [])
    except Exception as e:
        logger.error(f"Prometheus query failed for '{query}': {e}")
        return []


def collect_gpu_snapshot():
    metrics = {
        "temperature":  fetch_metric("dcgm_gpu_temp_celsius"),
        "utilization":  fetch_metric("dcgm_gpu_utilization_percent"),
        "mem_util":     fetch_metric("dcgm_mem_utilization_percent"),
        "power":        fetch_metric("dcgm_power_usage_watts"),
        "health_score": fetch_metric("dcgm_gpu_health_score"),
        "xid_errors":   fetch_metric("increase(dcgm_xid_errors_total[5m])"),
        "pcie_errors":  fetch_metric("increase(dcgm_pcie_replay_errors_total[5m])"),
        "sm_clock":     fetch_metric("dcgm_sm_clock_mhz"),
    }
    gpus = {}
    for metric_name, results in metrics.items():
        for result in results:
            gpu_id   = result["metric"].get("gpu_id", "unknown")
            gpu_name = result["metric"].get("gpu_name", "unknown")
            value    = float(result["value"][1])
            if gpu_id not in gpus:
                gpus[gpu_id] = {"gpu_id": gpu_id, "gpu_name": gpu_name}
            gpus[gpu_id][metric_name] = round(value, 2)
    return gpus


def build_prompt(gpus):
    gpu_data = json.dumps(gpus, indent=2)
    return f"""You are an expert GPU infrastructure reliability engineer at a large cloud provider.

Analyze the following real-time GPU telemetry snapshot and identify any anomalies or pre-failure indicators.

GPU TELEMETRY SNAPSHOT (collected at {datetime.utcnow().isoformat()}):
{gpu_data}

METRIC THRESHOLDS FOR REFERENCE:
- Temperature: normal <75C, warning 75-85C, critical >85C
- Health score: healthy >80, degraded 60-80, critical <60
- Memory utilization: normal <80%, warning 80-90%, critical >90%
- Xid errors in last 5 min: any value >0 is significant
- SM clock drop >200MHz from baseline (1410MHz) indicates throttling

For each GPU, respond ONLY with a valid JSON object in this exact format:
{{
  "gpu_id": {{
    "anomaly_score": <0-100 integer>,
    "status": "<normal|warning|critical>",
    "findings": "<one sentence summary>",
    "recommended_action": "<one sentence action if any>"
  }}
}}

anomaly_score guide: 0-20=normal, 21-50=minor anomaly, 51-80=warning, 81-100=critical
Respond with ONLY the JSON, no preamble, no markdown, no code fences."""


def call_gemini_api(prompt):
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set — skipping AI analysis")
        return {}
    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000}
            },
            timeout=30
        )
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return {}


def write_anomaly_scores(analysis, gpus):
    for gpu_id, result in analysis.items():
        gpu_name = gpus.get(gpu_id, {}).get("gpu_name", "unknown")
        score    = result.get("anomaly_score", 0)
        status   = result.get("status", "unknown")
        AI_ANOMALY_SCORE.labels(gpu_id=gpu_id, gpu_name=gpu_name).set(score)
        AI_ANALYSIS_TIMESTAMP.labels(gpu_id=gpu_id).set(time.time())
        logger.info(f"GPU {gpu_id} ({gpu_name}) | status={status} | anomaly_score={score} | {result.get('findings', '')}")
        if status == "critical":
            logger.warning(f"[CRITICAL] GPU {gpu_id} — Action: {result.get('recommended_action', 'N/A')}")


def run_analysis_loop():
    logger.info(f"AI anomaly detector started. Analysis every {ANALYSIS_INTERVAL}s")
    logger.info(f"Prometheus: {PROMETHEUS_URL}")
    logger.info(f"Gemini API key: {'set' if GEMINI_API_KEY else 'NOT SET'}")
    while True:
        try:
            logger.info("Collecting GPU snapshot from Prometheus...")
            gpus = collect_gpu_snapshot()
            if not gpus:
                logger.warning("No GPU data from Prometheus — is the simulator running?")
                time.sleep(ANALYSIS_INTERVAL)
                continue
            logger.info(f"Analysing {len(gpus)} GPUs with Gemini API...")
            prompt   = build_prompt(gpus)
            analysis = call_gemini_api(prompt)
            if analysis:
                write_anomaly_scores(analysis, gpus)
            else:
                logger.warning("No analysis returned from Gemini API")
        except Exception as e:
            logger.error(f"Analysis loop error: {e}")
        time.sleep(ANALYSIS_INTERVAL)


if __name__ == "__main__":
    start_http_server(METRICS_PORT)
    logger.info(f"AI metrics server on :{METRICS_PORT}")
    run_analysis_loop()
