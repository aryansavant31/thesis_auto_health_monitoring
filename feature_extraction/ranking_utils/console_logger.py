import sys
import os
import warnings
from datetime import datetime
from contextlib import contextmanager
from io import StringIO
import subprocess
import tempfile

class ConsoleLogger:
    def __init__(self):
        self.log_buffer = StringIO()
        self.original_stdout = None
        self.original_stderr = None
        self.original_warning_showwarning = None
        self.start_time = None
        self.end_time = None
        
    @contextmanager
    def capture_output(self):
        """Context manager to capture ALL output including low-level stderr"""
        
        self.start_time = datetime.now()
        
        # Create temporary files to capture low-level output
        temp_stdout = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        temp_stderr = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        
        class TeeOutput:
            def __init__(self, buffer_obj, terminal_obj, temp_file=None):
                self.buffer = buffer_obj
                self.terminal = terminal_obj
                self.temp_file = temp_file
                
            def write(self, message):
                self.terminal.write(message)
                self.buffer.write(message)
                if self.temp_file:
                    self.temp_file.write(message)
                    self.temp_file.flush()
                
            def flush(self):
                self.terminal.flush()
                self.buffer.flush()
                if self.temp_file:
                    self.temp_file.flush()
        
        # Custom warning handler
        def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
            warning_msg = f"{filename}:{lineno}: {category.__name__}: {message}\n"
            self.original_stderr.write(warning_msg)
            self.log_buffer.write(warning_msg)
            temp_stderr.write(warning_msg)
            temp_stderr.flush()
        
        # Store originals
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.original_warning_showwarning = warnings.showwarning
        
        # Duplicate file descriptors to capture low-level output
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)
        
        try:
            # Redirect file descriptors to temp files
            os.dup2(temp_stdout.fileno(), 1)
            os.dup2(temp_stderr.fileno(), 2)
            
            # Create tee objects
            tee_stdout = TeeOutput(self.log_buffer, self.original_stdout, temp_stdout)
            tee_stderr = TeeOutput(self.log_buffer, self.original_stderr, temp_stderr)
            
            # Redirect Python streams
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            warnings.showwarning = custom_warning_handler
            
            yield self
            
        finally:
            self.end_time = datetime.now()
            
            # Restore file descriptors
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
            
            # Restore Python streams
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            warnings.showwarning = self.original_warning_showwarning
            
            # Read any additional output from temp files
            temp_stdout.seek(0)
            temp_stderr.seek(0)
            
            additional_stdout = temp_stdout.read()
            additional_stderr = temp_stderr.read()
            
            if additional_stdout:
                self.log_buffer.write(additional_stdout)
            if additional_stderr:
                self.log_buffer.write(additional_stderr)
            
            # Clean up temp files
            temp_stdout.close()
            temp_stderr.close()
            os.unlink(temp_stdout.name)
            os.unlink(temp_stderr.name)
    
    def save_to_file(self, log_file_path, script_name, base_name):
        """Save the captured output to a file"""
        
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
            log_file.write(f"Base Name: {base_name}\n")
            log_file.write(f"Start Time: {start_time_str}\n")
            log_file.write(f"End Time: {end_time_str}\n")
            log_file.write("=" * 50 + "\n\n")
            
            # Write captured output
            log_file.write(self.log_buffer.getvalue())
            
            # Write footer
            log_file.write(f"\n\n=== EXECUTION COMPLETED ===\n")
            log_file.write(f"Log saved at: {end_time_str}\n")
        
        print(f"\nConsole output saved to: {log_file_path}")
    
    def get_output(self):
        """Get the captured output as a string"""
        return self.log_buffer.getvalue()
    
    def clear_buffer(self):
        """Clear the captured output buffer"""
        self.log_buffer = StringIO()
        self.start_time = None
        self.end_time = None