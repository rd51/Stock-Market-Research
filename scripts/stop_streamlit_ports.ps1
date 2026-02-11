$ports = @(8501,8502)
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conns) {
        $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($thePid in $pids) {
            try {
                Stop-Process -Id $thePid -Force -ErrorAction Stop
                Write-Output ("Stopped {0} on port {1}" -f $thePid, $port)
            } catch {
                Write-Output ("Failed to stop {0} on port {1}: {2}" -f $thePid, $port, ${_})
            }
        }
    } else {
        Write-Output "No process on port $port"
    }
}
