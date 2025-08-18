#!/usr/bin/env python3
"""
Monitoring script để theo dõi training progress
"""

import time
from pathlib import Path

def monitor_training():
    """Monitor training logs"""
    log_dir = Path("runs/train/trash_safe")
    
    print("🔍 Monitoring training progress...")
    print("💡 Ctrl+C để thoát monitor")
    
    try:
        while True:
            # Check if training started
            if log_dir.exists():
                results_file = log_dir / "results.csv"
                if results_file.exists():
                    # Read last few lines of results
                    with open(results_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # Header + at least 1 epoch
                            last_line = lines[-1].strip()
                            if last_line:
                                parts = last_line.split(',')
                                if len(parts) >= 10:
                                    epoch = parts[0].strip()
                                    train_loss = parts[1].strip() if len(parts) > 1 else 'N/A'
                                    val_loss = parts[2].strip() if len(parts) > 2 else 'N/A'
                                    map50 = parts[7].strip() if len(parts) > 7 else 'N/A'
                                    
                                    print(f"📊 Epoch {epoch}: Train Loss={train_loss}, Val Loss={val_loss}, mAP50={map50}")
            
            # Check GPU usage
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split(',')
                    if len(gpu_info) >= 3:
                        mem_used = gpu_info[0].strip()
                        mem_total = gpu_info[1].strip()
                        gpu_util = gpu_info[2].strip()
                        print(f"🖥️  GPU: {mem_used}MB/{mem_total}MB ({gpu_util}% utilization)")
            except:
                pass
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n✋ Monitor stopped")
    except Exception as e:
        print(f"❌ Monitor error: {e}")

if __name__ == "__main__":
    monitor_training()
