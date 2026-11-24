"""
FALLBACK ONLY - use if the VM's /mark calls do not reach victim_server.py.
Run in VS Code (Windows) from the project folder, then press Enter at the exact start/end of each phase on the VM:
    python stageB\\phase_logger.py
Writes attack_log.csv with the SAME columns as victim_server.py (epoch,phase,state). Never run it while victim_server.py is also logging.
"""
import csv, os, time
LOG = "attack_log.csv"
new = not os.path.exists(LOG)
f = open(LOG, "a", newline=""); w = csv.writer(f)
if new: w.writerow(["epoch", "phase", "state"])
while True:
    name = input("phase name (e.g. benign_1, portscan_2, dos_syn_3; blank to quit): ").strip()
    if not name: break
    input(f"  Press Enter the moment '{name}' STARTS on the VM... ")
    w.writerow([f"{time.time():.3f}", name, "start"]); f.flush()
    input("  Press Enter the moment it ENDS... ")
    w.writerow([f"{time.time():.3f}", name, "end"]); f.flush()
f.close()