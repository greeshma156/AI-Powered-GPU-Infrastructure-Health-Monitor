# AI-Powered GPU Infrastructure Health Monitor

> Real-time GPU fleet observability platform with AI-driven anomaly detection and automated recovery — built with Prometheus, Grafana, ELK, and the Gemini API.

---

## Overview

This project simulates a production-grade GPU infrastructure monitoring system as used in large-scale cloud data centers. It generates DCGM-style GPU telemetry, detects pre-failure indicators using AI, visualises fleet health in real time, and automatically recovers from hardware faults — all without manual intervention.

Built to demonstrate expertise in:
- Infrastructure observability (Prometheus, Grafana, ELK)
- GPU telemetry and diagnostics (DCGM-style metrics, Xid errors)
- AI-powered anomaly detection (Gemini API)
- Container orchestration (Docker, Kubernetes-ready)
- Automated recovery workflows
- CI/CD pipelines (GitHub Actions)

---

## Architecture

```
GPU Telemetry Simulator (Python)
        │
        ▼
Prometheus Exporter (:8000)
        │
        ▼
Prometheus (:9090) ──────────────────────────────────┐
        │                                             │
        ▼                                             ▼
AI Anomaly Detector                           Grafana (:3000)
  (Gemini API)                              (11-panel dashboard)
        │
        ▼
Anomaly Score → written back to Prometheus
        │
        ▼
Recovery Agent
  (auto-quarantine on score > 80)
        │
        ▼
Alertmanager (:9093)
```

---

## Metrics Exposed

| Metric | Description |
|--------|-------------|
| `dcgm_gpu_temp_celsius` | GPU temperature |
| `dcgm_gpu_utilization_percent` | Compute utilisation |
| `dcgm_mem_utilization_percent` | Memory utilisation |
| `dcgm_mem_used_mb` | Memory used (MB) |
| `dcgm_power_usage_watts` | Power draw |
| `dcgm_sm_clock_mhz` | SM clock speed |
| `dcgm_gpu_health_score` | Simulated health score (0–100) |
| `dcgm_xid_errors_total` | Xid error counter |
| `dcgm_pcie_replay_errors_total` | PCIe replay errors |
| `ai_anomaly_score` | AI-computed anomaly score (0–100) |
| `ai_last_analysis_timestamp` | Timestamp of last AI analysis |

---

## Stack

| Component | Purpose |
|-----------|---------|
| Python | GPU telemetry simulator + Prometheus exporter |
| Prometheus | Metrics scraping and storage |
| Grafana | Real-time dashboards (11 panels) |
| Alertmanager | Alert routing by severity |
| Gemini API | AI anomaly detection and health scoring |
| ELK Stack | Diagnostic event log storage and search |
| Docker Compose | Full stack orchestration |
| GitHub Actions | CI/CD — lint, test, Docker build |

---

## Quick Start

### Prerequisites
- Docker Desktop (4GB+ RAM allocated)
- Python 3.11+
- Gemini API key (free at https://aistudio.google.com/apikey)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/gpu-health-monitor
cd gpu-health-monitor
echo "GEMINI_API_KEY=your_key_here" > .env
docker compose up -d
curl http://localhost:8000/metrics | grep dcgm_gpu_temp
```

### Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Alertmanager | http://localhost:9093 | — |
| Metrics endpoint | http://localhost:8000/metrics | — |
| AI detector metrics | http://localhost:8001/metrics | — |

---

## How the AI Detection Works

1. Every 120 seconds the detector queries Prometheus for all GPU metrics
2. It builds a structured prompt with the telemetry snapshot and sends it to the Gemini API
3. Gemini analyses the combination of metrics holistically and returns a JSON response with anomaly score (0–100), status, findings, and recommended action per GPU
4. The anomaly score is written back to Prometheus as `ai_anomaly_score`
5. Grafana displays the score in real time
6. If any GPU scores above 80, the recovery agent automatically quarantines it

---

## Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| GPUHighTemperature | temp > 80°C for 2m | warning |
| GPUCriticalTemperature | temp > 90°C for 1m | critical |
| GPULowHealthScore | health < 60 for 3m | warning |
| GPUCriticalHealthScore | health < 30 for 1m | critical |
| GPUXidErrors | any Xid error in 5m | critical |
| GPUHighMemoryUtilization | mem > 90% for 5m | warning |

---

## Running Tests

```bash
pip install pytest prometheus_client requests
pytest tests/ -v
```

---

## CI/CD Pipeline

Every push to main triggers:
1. Lint — flake8 on all Python files
2. Unit tests — pytest
3. Docker builds — all 3 images built and verified
4. Integration test — full stack verified end to end

---

## License

MIT
