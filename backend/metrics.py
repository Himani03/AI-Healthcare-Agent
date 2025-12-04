import json
import os
import time
import pandas as pd
from datetime import datetime

METRICS_FILE = "metrics_log.json"

class MetricsTracker:
    def __init__(self):
        self.file_path = METRICS_FILE
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)

    def log_inference(self, module_name, duration_seconds, success=True, error_msg=None):
        """Log a single inference event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "module": module_name,
            "duration": round(duration_seconds, 4),
            "success": success,
            "error": str(error_msg) if error_msg else None,
            "type": "inference"  # Distinguish from feedback
        }
        self._append_event(event)

    def log_feedback(self, module_name, is_positive: bool):
        """Log user feedback (thumbs up/down)"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "module": module_name,
            "feedback": "positive" if is_positive else "negative",
            "type": "feedback"
        }
        self._append_event(event)

    def _append_event(self, event):
        """Helper to append event to JSON file"""
        try:
            with open(self.file_path, 'r+') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                
                data.append(event)
                f.seek(0)
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"❌ Error logging metrics: {e}")

    def get_metrics_dataframe(self):
        """Return metrics as a Pandas DataFrame"""
        if not os.path.exists(self.file_path):
            return pd.DataFrame()
        
        try:
            df = pd.read_json(self.file_path)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"❌ Error reading metrics: {e}")
            return pd.DataFrame()

# Singleton
metrics_tracker = MetricsTracker()
