import statistics

class AccidentDetector:
    def __init__(self):
        self.history = {} # Store aspect ratios history for IDs

    def detect_accident(self, persons):
        """
        persons: List of dicts with 'id', 'bbox', 'landmarks' (optional)
        bbox assumed to be [x1, y1, x2, y2]
        """
        alerts = []
        for p in persons:
            pid = p['id']
            bbox = p['bbox']
            
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            if h == 0: continue
            
            aspect_ratio = h / w # > 1 means tall/standing, < 1 means wide/laying
            
            # Simple Fall Detection: Sudden transition from Standing to Lying
            prev_ratios = self.history.get(pid, [])
            
            if len(prev_ratios) > 5:
                avg_prev = statistics.mean(prev_ratios[-10:]) # Average of last 10 frames
                
                # Check for Fall
                # Was standing (avg > 1.2) and now lying (current < 0.8)
                if avg_prev > 1.2 and aspect_ratio < 0.8:
                    alerts.append(f"ACCIDENT: Fall detected for Person ID {pid}")
            
            # Update history
            if pid not in self.history:
                self.history[pid] = []
            self.history[pid].append(aspect_ratio)
            if len(self.history[pid]) > 20:
                self.history[pid].pop(0)
                
        return alerts
