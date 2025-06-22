"""
ext_system_monitor.py - System Health Monitoring and Validation

This extension monitors system health, validates integrity, and reports
anomalies.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import psutil
from collections import deque

from extensions.base import GyroExtension


class ext_SystemMonitor(GyroExtension):
    """
    System health monitoring and validation.
    FOOTPRINT: 50-100 bytes (monitoring cache)
    MAPPING: Tracks and reports system health and diagnostics
    """

    def __init__(self):
        """Initialize system monitor."""
        # Health metrics history
        self._health_history = deque(maxlen=100)

        # Performance metrics
        self._performance_metrics = {
            "cpu_usage": deque(maxlen=60),
            "memory_usage": deque(maxlen=60),
            "operation_latency": deque(maxlen=100),
        }

        # Anomaly detection
        self._anomalies = deque(maxlen=50)
        self._anomaly_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "latency_ms": 100.0,
            "error_rate": 0.05,
        }

        # System state
        self._system_state = {
            "status": "healthy",
            "uptime_start": time.time(),
            "last_check": time.time(),
            "check_interval": 10.0,  # seconds
        }

        # Validation results cache
        self._validation_cache = {}
        self._last_validation = 0

    def get_health_report(
        self, phase: int, nav_log_size: int, knowledge_id: str, session_id: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive health report.

        Args:
            phase: Current phase
            nav_log_size: Navigation log size
            knowledge_id: Current knowledge ID
            session_id: Current session ID

        Returns:
            Health report dictionary
        """
        current_time = time.time()

        # Collect system metrics
        metrics = self._collect_system_metrics()

        # Check for anomalies
        anomalies = self._check_anomalies(metrics)

        # Build health report
        report = {
            "timestamp": current_time,
            "status": self._system_state["status"],
            "uptime": current_time - self._system_state["uptime_start"],
            "system_metrics": metrics,
            "gyro_state": {
                "phase": phase,
                "nav_log_size": nav_log_size,
                "knowledge_id": knowledge_id,
                "session_id": session_id,
            },
            "anomalies": anomalies,
            "performance": self._get_performance_summary(),
        }

        # Update history
        self._health_history.append(report)
        self._system_state["last_check"] = current_time

        return report

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "timestamp": time.time(),
            }

            # Update performance history
            self._performance_metrics["cpu_usage"].append(cpu_percent)
            self._performance_metrics["memory_usage"].append(memory.percent)

        except Exception:
            # Fallback if psutil not available
            metrics = {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "memory_available_mb": 0.0,
                "timestamp": time.time(),
            }

        return metrics

    def _check_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for system anomalies."""
        anomalies = []

        # CPU anomaly
        if metrics["cpu_percent"] > self._anomaly_thresholds["cpu_percent"]:
            anomaly = {
                "type": "high_cpu",
                "severity": "warning",
                "value": metrics["cpu_percent"],
                "threshold": self._anomaly_thresholds["cpu_percent"],
                "timestamp": metrics["timestamp"],
            }
            anomalies.append(anomaly)
            self._anomalies.append(anomaly)

        # Memory anomaly
        if metrics["memory_percent"] > self._anomaly_thresholds["memory_percent"]:
            anomaly = {
                "type": "high_memory",
                "severity": "warning",
                "value": metrics["memory_percent"],
                "threshold": self._anomaly_thresholds["memory_percent"],
                "timestamp": metrics["timestamp"],
            }
            anomalies.append(anomaly)
            self._anomalies.append(anomaly)

        # Update system status
        if anomalies:
            self._system_state["status"] = "degraded"
        else:
            self._system_state["status"] = "healthy"

        return anomalies

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {}

        # CPU statistics
        if self._performance_metrics["cpu_usage"]:
            cpu_values = list(self._performance_metrics["cpu_usage"])
            summary["cpu"] = {
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            }

        # Memory statistics
        if self._performance_metrics["memory_usage"]:
            mem_values = list(self._performance_metrics["memory_usage"])
            summary["memory"] = {
                "average": sum(mem_values) / len(mem_values),
                "max": max(mem_values),
                "min": min(mem_values),
            }

        # Latency statistics
        if self._performance_metrics["operation_latency"]:
            lat_values = list(self._performance_metrics["operation_latency"])
            summary["latency"] = {
                "average": sum(lat_values) / len(lat_values),
                "max": max(lat_values),
                "min": min(lat_values),
                "p95": sorted(lat_values)[int(len(lat_values) * 0.95)],
            }

        return summary

    def ext_record_operation_latency(self, latency_ms: float) -> None:
        """
        Record operation latency for monitoring.

        Args:
            latency_ms: Operation latency in milliseconds
        """
        self._performance_metrics["operation_latency"].append(latency_ms)

        # Check for latency anomaly
        if latency_ms > self._anomaly_thresholds["latency_ms"]:
            anomaly = {
                "type": "high_latency",
                "severity": "info",
                "value": latency_ms,
                "threshold": self._anomaly_thresholds["latency_ms"],
                "timestamp": time.time(),
            }
            self._anomalies.append(anomaly)

    def ext_validate_system_integrity(self) -> Tuple[bool, Dict[str, bool]]:
        """
        Perform comprehensive system validation.

        Returns:
            Tuple of (overall_valid, detailed_results)
        """
        current_time = time.time()

        # Use cache if recent
        if current_time - self._last_validation < 60:  # 1 minute cache
            if "results" in self._validation_cache:
                return self._validation_cache["overall"], self._validation_cache["results"]

        # Perform validation checks
        results = {
            "memory_bounds": self._validate_memory_bounds(),
            "performance_bounds": self._validate_performance_bounds(),
            "anomaly_rate": self._validate_anomaly_rate(),
            "system_resources": self._validate_system_resources(),
        }

        overall = all(results.values())

        # Cache results
        self._validation_cache = {"overall": overall, "results": results, "timestamp": current_time}
        self._last_validation = current_time

        return overall, results

    def _validate_memory_bounds(self) -> bool:
        """Validate memory usage is within bounds."""
        if not self._performance_metrics["memory_usage"]:
            return True

        avg_memory = sum(self._performance_metrics["memory_usage"]) / len(
            self._performance_metrics["memory_usage"]
        )
        return avg_memory < 90.0  # Less than 90% average

    def _validate_performance_bounds(self) -> bool:
        """Validate performance is within acceptable bounds."""
        if not self._performance_metrics["operation_latency"]:
            return True

        avg_latency = sum(self._performance_metrics["operation_latency"]) / len(
            self._performance_metrics["operation_latency"]
        )
        return avg_latency < 50.0  # Less than 50ms average

    def _validate_anomaly_rate(self) -> bool:
        """Validate anomaly rate is acceptable."""
        if not self._anomalies:
            return True

        # Check anomaly rate in last hour
        current_time = time.time()
        recent_anomalies = [a for a in self._anomalies if current_time - a["timestamp"] < 3600]

        # Less than 10 anomalies per hour
        return len(recent_anomalies) < 10

    def _validate_system_resources(self) -> bool:
        """Validate system has sufficient resources."""
        try:
            memory = psutil.virtual_memory()
            # At least 100MB available
            return memory.available > 100 * 1024 * 1024
        except Exception:
            return True  # Assume OK if can't check

    # --- GyroExtension Interface Implementation ---

    def get_extension_name(self) -> str:
        return "ext_system_monitor"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        # Monitoring cache and metrics
        return 100

    def get_learning_state(self) -> Dict[str, Any]:
        """System patterns and thresholds."""
        return {
            "anomaly_thresholds": self._anomaly_thresholds.copy(),
            "performance_summary": self._get_performance_summary(),
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Current monitoring state."""
        return {
            "system_state": self._system_state.copy(),
            "recent_anomalies": list(self._anomalies)[-10:],
        }

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore thresholds."""
        if "anomaly_thresholds" in state:
            self._anomaly_thresholds.update(state["anomaly_thresholds"])

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore monitoring state."""
        if "system_state" in state:
            self._system_state.update(state["system_state"])
