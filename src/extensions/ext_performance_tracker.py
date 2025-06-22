"""
ext_performance_tracker.py - Performance Metrics Tracking

This extension tracks detailed performance metrics for all system operations,
providing insights into system efficiency and bottlenecks.
"""

from typing import Dict, Any, Optional
import time
from collections import deque, defaultdict
import statistics

from extensions.base import GyroExtension


class ext_PerformanceTracker(GyroExtension):
    """
    Tracks extension and system performance metrics.
    FOOTPRINT: 100-200 bytes (metrics cache)
    MAPPING: Captures and logs performance statistics
    """

    def __init__(self):
        """Initialize performance tracker."""
        # Operation timing history
        self._operation_timings = defaultdict(lambda: deque(maxlen=1000))

        # Current operation stack for nested timing
        self._operation_stack = []

        # Performance metrics by component
        self._component_metrics = defaultdict(
            lambda: {
                "call_count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "errors": 0,
            }
        )

        # System-wide metrics
        self._system_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_resonances": 0,
            "resonance_rate": 0.0,
            "operations_per_second": 0.0,
            "last_update": time.time(),
        }

        # Throughput tracking
        self._throughput_window = deque(maxlen=60)  # 1-minute window
        self._last_throughput_update = time.time()

        # Memory usage tracking
        self._memory_snapshots = deque(maxlen=100)

        # Performance alerts
        self._performance_alerts = deque(maxlen=50)
        self._alert_thresholds = {
            "operation_time_ms": 100.0,
            "memory_growth_mb": 10.0,
            "error_rate": 0.05,
        }

    def start_operation(self, operation_name: Optional[str] = None) -> None:
        """
        Start timing an operation.

        Args:
            operation_name: Name of the operation (default: 'navigation_cycle')
        """
        if operation_name is None:
            operation_name = "navigation_cycle"

        operation = {
            "name": operation_name,
            "start_time": time.perf_counter(),
            "start_memory": self._get_memory_usage(),
        }

        self._operation_stack.append(operation)
        self._system_metrics["total_operations"] += 1

    def end_operation(self, resonated: bool = False, error: Optional[Exception] = None) -> float:
        """
        End timing an operation.

        Args:
            resonated: Whether resonance occurred
            error: Exception if operation failed

        Returns:
            Operation duration in milliseconds
        """
        if not self._operation_stack:
            return 0.0

        operation = self._operation_stack.pop()
        end_time = time.perf_counter()
        duration_ms = (end_time - operation["start_time"]) * 1000

        # Record timing
        operation_name = operation["name"]
        self._operation_timings[operation_name].append(duration_ms)

        # Update component metrics
        metrics = self._component_metrics[operation_name]
        metrics["call_count"] += 1
        metrics["total_time"] += duration_ms
        metrics["min_time"] = min(metrics["min_time"], duration_ms)
        metrics["max_time"] = max(metrics["max_time"], duration_ms)

        if error:
            metrics["errors"] += 1
            self._system_metrics["failed_operations"] += 1
        else:
            self._system_metrics["successful_operations"] += 1

        if resonated:
            self._system_metrics["total_resonances"] += 1

        # Check for performance alerts
        if duration_ms > self._alert_thresholds["operation_time_ms"]:
            self._record_alert(
                "slow_operation",
                {
                    "operation": operation_name,
                    "duration_ms": duration_ms,
                    "threshold": self._alert_thresholds["operation_time_ms"],
                },
            )

        # Update throughput
        self._update_throughput()

        return duration_ms

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _update_throughput(self) -> None:
        """Update operations per second metric."""
        current_time = time.time()

        # Add current timestamp
        self._throughput_window.append(current_time)

        # Remove old timestamps (older than 60 seconds)
        cutoff = current_time - 60
        while self._throughput_window and self._throughput_window[0] < cutoff:
            self._throughput_window.popleft()

        # Calculate operations per second
        if len(self._throughput_window) > 1:
            time_span = self._throughput_window[-1] - self._throughput_window[0]
            if time_span > 0:
                self._system_metrics["operations_per_second"] = (
                    len(self._throughput_window) / time_span
                )

        # Update resonance rate
        if self._system_metrics["total_operations"] > 0:
            self._system_metrics["resonance_rate"] = (
                self._system_metrics["total_resonances"] / self._system_metrics["total_operations"]
            )

    def _record_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Record a performance alert."""
        alert = {"type": alert_type, "timestamp": time.time(), "details": details}
        self._performance_alerts.append(alert)

    def ext_get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Operation statistics
        """
        if operation_name not in self._operation_timings:
            return {"error": "Operation not found"}

        timings = list(self._operation_timings[operation_name])
        if not timings:
            return {"error": "No timing data"}

        metrics = self._component_metrics[operation_name]

        return {
            "operation": operation_name,
            "call_count": metrics["call_count"],
            "average_time_ms": metrics["total_time"] / metrics["call_count"],
            "min_time_ms": metrics["min_time"],
            "max_time_ms": metrics["max_time"],
            "median_time_ms": statistics.median(timings),
            "p95_time_ms": (
                sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings)
            ),
            "error_count": metrics["errors"],
            "error_rate": (
                metrics["errors"] / metrics["call_count"] if metrics["call_count"] > 0 else 0
            ),
        }

    def ext_get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        # Calculate component breakdown
        component_breakdown = {}
        total_time = sum(m["total_time"] for m in self._component_metrics.values())

        if total_time > 0:
            for component, metrics in self._component_metrics.items():
                component_breakdown[component] = {
                    "percentage": (metrics["total_time"] / total_time) * 100,
                    "calls": metrics["call_count"],
                }

        # Sort by percentage
        sorted_components = sorted(
            component_breakdown.items(), key=lambda x: x[1]["percentage"], reverse=True
        )[
            :10
        ]  # Top 10

        return {
            "system_metrics": self._system_metrics.copy(),
            "top_components": dict(sorted_components),
            "recent_alerts": list(self._performance_alerts)[-10:],
            "memory_usage_mb": self._get_memory_usage(),
        }

    def ext_get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "summary": self.ext_get_system_performance(),
            "operations": {},
            "bottlenecks": [],
            "recommendations": [],
        }

        # Add operation details
        for op_name in self._operation_timings:
            report["operations"][op_name] = self.ext_get_operation_stats(op_name)

        # Identify bottlenecks
        bottlenecks = []
        for op_name, metrics in self._component_metrics.items():
            avg_time = (
                metrics["total_time"] / metrics["call_count"] if metrics["call_count"] > 0 else 0
            )
            if avg_time > 50:  # Operations taking >50ms on average
                bottlenecks.append(
                    {
                        "operation": op_name,
                        "average_time_ms": avg_time,
                        "impact": "high" if avg_time > 100 else "medium",
                    }
                )

        report["bottlenecks"] = sorted(
            bottlenecks, key=lambda x: x["average_time_ms"], reverse=True
        )

        # Generate recommendations
        if self._system_metrics["error_rate"] > 0.05:
            report["recommendations"].append(
                {
                    "type": "high_error_rate",
                    "message": f"Error rate is {self._system_metrics['error_rate']:.2%}. Investigate failing operations.",
                    "severity": "high",
                }
            )

        if self._system_metrics["operations_per_second"] < 10:
            report["recommendations"].append(
                {
                    "type": "low_throughput",
                    "message": f"Throughput is {self._system_metrics['operations_per_second']:.1f} ops/sec. Consider optimization.",
                    "severity": "medium",
                }
            )

        return report

    def ext_reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self._operation_timings.clear()
        self._component_metrics.clear()
        self._system_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_resonances": 0,
            "resonance_rate": 0.0,
            "operations_per_second": 0.0,
            "last_update": time.time(),
        }
        self._throughput_window.clear()
        self._performance_alerts.clear()

    # --- GyroExtension Interface Implementation ---

    def get_extension_name(self) -> str:
        return "ext_performance_tracker"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        # Metrics cache size
        footprint = 0
        footprint += len(self._operation_timings) * 50  # Timing deques
        footprint += len(self._component_metrics) * 40  # Metric dictionaries
        footprint += 100  # Fixed overhead
        return min(footprint, 200)  # Cap at 200 bytes

    def get_learning_state(self) -> Dict[str, Any]:
        """Performance patterns and statistics."""
        return {
            "component_metrics": dict(self._component_metrics),
            "alert_thresholds": self._alert_thresholds.copy(),
            "system_performance": self.ext_get_system_performance(),
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Current performance tracking state."""
        return {
            "system_metrics": self._system_metrics.copy(),
            "recent_alerts": list(self._performance_alerts)[-20:],
        }

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore performance patterns."""
        if "alert_thresholds" in state:
            self._alert_thresholds.update(state["alert_thresholds"])

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore current metrics."""
        if "system_metrics" in state:
            self._system_metrics.update(state["system_metrics"])
