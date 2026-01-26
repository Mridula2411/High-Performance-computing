import threading
import psutil
import time
import atexit
from functools import wraps
from .utils import create_evolution_plot, create_summary_table


# Global profiler instance to track all calls and ensure atexit is registered early
_global_profiler = None


def _get_global_profiler(interval=1):
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = CPUProfiler(interval=interval)
        # Register cleanup at module load time to avoid atexit registration errors
        atexit.register(_global_profiler._cleanup)
    return _global_profiler



class CPUProfiler:
    def __init__(self, interval=1):
        self.interval = interval
        cpu_count = psutil.cpu_count()
        self.cpu_count = cpu_count if cpu_count is not None else 1
        self.cpu_usage = []
        self.timestamps = []
        self._stop_event = threading.Event()
        self.thread = None
        self._is_running = False
        self._start_time = None  # Global timer, set once on first start
        self._warmed_optional = False  # Avoid late optional imports during shutdown

    
    def _profile_cpu(self):
        """Background thread function to profile CPU usage."""
        while not self._stop_event.is_set():
            usage = psutil.cpu_percent(interval=self.interval, percpu=True)
            timestamp = time.time() - self._start_time
            self.cpu_usage.append(usage)
            self.timestamps.append(timestamp)

    def start(self):
        """Start the CPU profiling in a background thread."""
        if self._is_running:
            return

        if not self._warmed_optional:
            self._warm_optional_imports()
        
        # Initialize global timer on first start
        if self._start_time is None:
            self._start_time = time.time()
        
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._profile_cpu, daemon=True)
        self.thread.start()
        self._is_running = True

    def stop(self):
        """Stop the CPU profiling."""
        if not self._is_running or self.thread is None:
            return
        self._stop_event.set()
        self.thread.join(timeout=2)
        self._is_running = False


    def _warm_optional_imports(self):
        """Eagerly import optional deps to avoid atexit registration during shutdown."""
        if self._warmed_optional:
            return
        try:
            import IPython  # noqa: F401
        except Exception:
            pass
        self._warmed_optional = True


    def generate_report(self):
        core_labels = [f"Core {i}" for i in range(self.cpu_count)]
        # Call the generalized plotter
        create_evolution_plot(
            timestamps=self.timestamps,
            data=self.cpu_usage,
            title="CPU Utilization Per Core",
            y_label="Usage (%)",
            labels=core_labels
        )

        table_str = create_summary_table(
            timestamps=self.timestamps, 
            data=self.cpu_usage, 
            headers=core_labels
        )
        print(table_str)
        return table_str

    def _cleanup(self):
        """Called at program exit to stop profiler and generate report."""
        self.stop()
        if self.cpu_usage:  # Only generate report if data was collected
            self.generate_report()
    


def profile_cpu(interval=1):
    """Decorator to profile CPU usage of a function across all its calls."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = _get_global_profiler(interval=interval)
            profiler.start()
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.stop()
            return result
        return wrapper
    return decorator