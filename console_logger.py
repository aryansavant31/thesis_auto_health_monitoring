import sys
import os
from datetime import datetime
from contextlib import contextmanager
from io import StringIO
import tempfile
import platform
import psutil
import torch

class ConsoleLogger:
    def __init__(self):
        self.log_buffer = StringIO()
        self.temp_file = None
        self.temp_file_path = None
        self.original_stdout = None
        self.original_stderr = None
        self.start_time = None
        self.end_time = None
        self.use_temp_file = False
        
    @contextmanager
    def capture_output(self, use_temp_file=False):
        """
        Context manager to capture all terminal output

        Parameters
        ----------
            use_temp_file: bool
                If True, uses temporary file instead of memory buffer
        """
        
        self.use_temp_file = use_temp_file
        self.start_time = datetime.now()
        
        if use_temp_file:
            # Create temporary file for large outputs
            self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
            self.temp_file_path = self.temp_file.name
        
        class TeeOutput:
            def __init__(self, buffer_obj, terminal_obj, temp_file=None):
                self.buffer = buffer_obj
                self.terminal = terminal_obj
                self.temp_file = temp_file
                
            def write(self, message):
                self.terminal.write(message)
                if self.temp_file:
                    self.temp_file.write(message)
                    self.temp_file.flush()
                else:
                    self.buffer.write(message)
                
            def flush(self):
                self.terminal.flush()
                if self.temp_file:
                    self.temp_file.flush()
                else:
                    self.buffer.flush()
        
        # Store original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create tee objects
        if use_temp_file:
            tee_stdout = TeeOutput(None, self.original_stdout, self.temp_file)
            tee_stderr = TeeOutput(None, self.original_stderr, self.temp_file)
        else:
            tee_stdout = TeeOutput(self.log_buffer, self.original_stdout)
            tee_stderr = TeeOutput(self.log_buffer, self.original_stderr)
        
        try:
            # Redirect stdout and stderr
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            
            yield self
            
        finally:
            self.end_time = datetime.now()
            # Restore original stdout and stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            if self.temp_file:
                self.temp_file.close()

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
    
    def save_to_file(self, log_file_path, script_name, base_name=None):
        """
        Save the captured output to a file
        """
        
        if not self.start_time:
            raise ValueError("No output has been captured yet. Use capture_output() first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Format timestamps
        start_time_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = self.end_time.strftime("%Y-%m-%d %H:%M:%S") if self.end_time else "Still running"
        
        # Write to file
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            # Write header
            log_file.write(f"=== SCRIPT EXECUTION LOG ===\n")
            log_file.write(f"Script: {script_name}\n")
            if base_name:
                log_file.write(f"Base Name: {base_name}\n")
            log_file.write(f"Start Time: {start_time_str}\n")
            log_file.write(f"End Time: {end_time_str}\n")

            # add system specs
            log_file.write("\n" + self._get_system_info() + "\n")
            log_file.write(f"OS: {platform.system()} {platform.release()} ({platform.version()})\n")
            log_file.write(f"\nPython Version: {sys.version}\n")
            
            log_file.write("=" * 120 + "\n\n")
            
            # Write captured output
            if self.use_temp_file and self.temp_file_path:
                # Copy from temp file
                with open(self.temp_file_path, 'r', encoding='utf-8') as temp_file:
                    log_file.write(temp_file.read())
                # Clean up temp file
                os.unlink(self.temp_file_path)
            else:
                # Copy from memory buffer
                log_file.write(self.log_buffer.getvalue())
            
            # Write footer
            log_file.write(f"\n\n=== EXECUTION COMPLETED ===\n")
            log_file.write(f"Log saved at: {end_time_str}\n")
        
        print(f"\nConsole output saved to: {log_file_path}")
    
    def get_output(self):
        """Get the captured output as a string"""
        if self.use_temp_file and self.temp_file_path:
            with open(self.temp_file_path, 'r', encoding='utf-8') as temp_file:
                return temp_file.read()
        return self.log_buffer.getvalue()
    
    def clear_buffer(self):
        """Clear the captured output buffer"""
        self.log_buffer = StringIO()
        self.start_time = None
        self.end_time = None
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)