# Run from the project folder BEFORE every new capture run:
#   powershell -ExecutionPolicy Bypass -File stageB\new_run.ps1
# Moves the previous run's logs/dumps into old_runs\<timestamp>\ so runs never mix.
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$dest = "old_runs\$stamp"; New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($f in "attack_log.csv","live_features.csv","live_results.csv","labelled_live_results.csv","live_capture_confusion.png") {
  if (Test-Path $f) { Move-Item $f $dest; Write-Host "archived $f" }
}
Write-Host "clean. archived to $dest"
