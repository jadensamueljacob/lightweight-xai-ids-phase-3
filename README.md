# Live network feed — setup & run order

Folder layout (put these next to your existing app.py, model.pkl, final_features.pkl, X_res.npy):
  app.py
  live_backend.py
  flow_sensor_replay.py
  flow_sensor_live.py
  rebuild_scaler.py
  pages/1_Live_Network_Feed.py      <- Streamlit auto-adds this as a second page, app.py is untouched

## One-time
pip install -U streamlit scikit-learn pandas numpy matplotlib lime shap psutil scapy
1. Put the 5 CSVs + feature_names.pkl + final_features.pkl + X_test_selected.npy + y_test.npy in one folder,
   edit CSV_DIR at the top of rebuild_scaler.py, run:  python rebuild_scaler.py
   -> must print "test labels identical: True | scaled features identical: True"
   -> creates scaler_15.json and replay_flows.csv (copy both next to app.py)
2. Fix CLASS_NAMES in app.py (see chat message) so it matches the "Class id -> name" line the script prints.

## Stage A - simulated live feed (Windows only, no VM)
Terminal 1:  streamlit run app.py      -> open "Live Network Feed" page (sidebar) -> click Start listener
Terminal 2:  python flow_sensor_replay.py --mode scenario

## Stage B - real packet capture
1. Install Npcap (https://npcap.com) with "WinPcap API-compatible mode" ticked.
2. ADMIN terminal:  python flow_sensor_live.py --list-ifaces
3. ADMIN terminal:  python flow_sensor_live.py --iface "<adapter name>" --host 127.0.0.1
   (sensor in a VM / on another PC: --host <IP of the PC running the dashboard>, and allow TCP 9009 in Windows Firewall)
4. Generate traffic on an ISOLATED lab network only (your own VM / your own machines):
   Benign : browse to http://<victim>:8000 (victim: python -m http.server 8000), curl in a loop, ping
   PortScan: nmap -sS -p 1-5000 <victim>
   DoS     : hping3 -S -p 8000 -i u2000 <victim>      (slowloris <victim> -p 8000 -s 100 for slow DoS)
