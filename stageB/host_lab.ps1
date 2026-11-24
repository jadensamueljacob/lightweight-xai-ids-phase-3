# Windows host helper. Run in an ADMIN PowerShell from this folder:
#   powershell -ExecutionPolicy Bypass -File host_lab.ps1 ips      # show VMware adapter IPs
#   powershell -ExecutionPolicy Bypass -File host_lab.ps1 add      # open victim ports
#   powershell -ExecutionPolicy Bypass -File host_lab.ps1 fwoff    # firewall OFF (needed for PortScan RSTs; lab only)
#   powershell -ExecutionPolicy Bypass -File host_lab.ps1 fwon     # firewall back ON
#   powershell -ExecutionPolicy Bypass -File host_lab.ps1 remove   # delete the rule
param([Parameter(Position=0)][string]$Action = "ips")

switch ($Action) {
  "ips"    { Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -like "*VMware*" } |
             Select-Object InterfaceAlias, IPAddress | Format-Table -AutoSize }
  "add"    { New-NetFirewallRule -DisplayName "xaiids-lab" -Direction Inbound -Protocol TCP `
             -LocalPort 8000,2121,2222 -Action Allow | Out-Null; Write-Host "rule added" }
  "remove" { Remove-NetFirewallRule -DisplayName "xaiids-lab" -ErrorAction SilentlyContinue; Write-Host "rule removed" }
  "fwoff"  { Set-NetFirewallProfile -Profile Domain,Private,Public -Enabled False; Write-Host "FIREWALL OFF - turn it back on after the lab" }
  "fwon"   { Set-NetFirewallProfile -Profile Domain,Private,Public -Enabled True;  Write-Host "firewall ON" }
  default  { Write-Host "unknown action" }
}
