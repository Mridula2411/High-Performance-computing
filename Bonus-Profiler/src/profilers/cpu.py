import threading
import psutil
import time
from functools import wraps
from .utils import create_evolution_plot, create_summary_table



class CPUProfiler:
    def __init__(self, interval=1):
        self.interval = interval
        cpu_count = psutil.cpu_count()
        self.cpu_count = cpu_count if cpu_count is not None else 1
        self.cpu_usage = []
        self.timestamps = []
        self._stop_event = threading.Event()

    
    def _profile_cpu(self):
        """Background thread function to profile CPU usage."""

        start_time = time.time()
        while not self._stop_event.is_set():
            usage = psutil.cpu_percent(interval=self.interval, percpu=True)
            timestamp = time.time() - start_time
            self.cpu_usage.append(usage)
            self.timestamps.append(timestamp)

    def start(self):
        """Start the CPU profiling in a background thread."""
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._profile_cpu)
        self.thread.start()

    def stop(self):
        """Stop the CPU profiling."""
        self._stop_event.set()
        self.thread.join()


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
    


def profile_cpu(interval=1):
    """Decorator to profile CPU usage of a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = CPUProfiler(interval=interval)
            profiler.start()
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.stop()
                report_str = profiler.generate_report()

                # maybe optionally save report_str to a file here or something if needed

            return result
        return wrapper
    return decorator