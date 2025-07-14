import sys
import io
import atexit
from pathlib import Path
import torch
import psutil
import platform
import datetime
from torch.utils.tensorboard import SummaryWriter

class StdoutToTensorBoard(io.StringIO):
    def __init__(self, log_dir="lightning_logs/full_logs", tag="Full Console Output"):
        super().__init__()

        caller_path = Path(sys._getframe(1).f_globals.get('__file__', '.')).parent
        resolved_log_dir = caller_path / log_dir
        resolved_log_dir.mkdir(parents=True, exist_ok=True)

        self._original_stdout = sys.stdout
        self._writer = SummaryWriter(log_dir=str(resolved_log_dir))
        self._tag = tag
        self._stopped = False

        # Gather system info
        self._system_info = self._get_system_info()

        # Write timestamp and system info at the top of the log
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write(f"Timestamp: {timestamp}\n")
        self.write("\n" + self._system_info + "\n"+"\n")

        sys.stdout = self
        atexit.register(self.cleanup)

    def _get_system_info(self):
        # CPU info
        cpu_name = platform.processor() or platform.uname().processor or "Unknown CPU"
        cpu_cores = psutil.cpu_count(logical=True) if psutil else "Unknown"
        cpu_info = f"CPU: {cpu_name} (Cores: {cpu_cores})"
        if psutil:
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    cpu_info += f", Max Frequency: {cpu_freq.max:.2f} MHz"
            except Exception:
                pass

        # GPU info
        if torch and torch.cuda.is_available():
            try:
                num_gpus = torch.cuda.device_count()
                gpu_infos = []
                for i in range(num_gpus):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_mem = f"{gpu_props.total_memory / (1024**3):.2f} GB"
                    gpu_infos.append(f"GPU {i}: {gpu_name}, Memory: {gpu_mem}")
                gpu_info = f"GPUs Detected: {num_gpus}\n" + "\n".join(gpu_infos)
            except Exception:
                gpu_info = "GPU: Error retrieving GPU info"
        else:
            gpu_info = "GPU: None detected"

        return f"{cpu_info}\n{gpu_info}"

    def write(self, text):
        if self._stopped:
            self._original_stdout.write(text)
            self._original_stdout.flush()
            return

        super().write(text)
        self._original_stdout.write(text)
        self._original_stdout.flush()

    def flush(self):
        super().flush()
        self._original_stdout.flush()

    def stop_logging(self):
        """Stop logging and restore stdout without closing the writer (for manual control)."""
        if not self._stopped:
            self._stopped = True
            self.restore()
            self.seek(0)
            full_log = self.read()
            self._writer.add_text(self._tag, f"```\n{full_log}\n```", global_step=0)
            self._writer.close()

    def cleanup(self):
        """Called at exit — only runs if stop_logging() wasn’t called manually."""
        if not self._stopped:
            self.stop_logging()

    def restore(self):
        if sys.stdout is self:
            sys.stdout = self._original_stdout