param([Parameter(Mandatory = $true)][string]$Root)
$ErrorActionPreference = 'Stop'
try {
    $rootPath = [IO.Path]::GetFullPath($Root)
    $enginePath = Join-Path $rootPath 'bin\basalt.exe'
    $engineArgument = '(?:^|\s)--engine-bin\s+(?:"' + [regex]::Escape($enginePath) + '"|'
    $engineArgument += [regex]::Escape($enginePath) + ')(?:\s|$)'
    $processes = @(Get-CimInstance Win32_Process)
    $roots = @($processes | Where-Object {
        $command = ([string]$_.CommandLine).Replace('/', '\')
        ($_.ExecutablePath -eq $enginePath -and $command -match '(?:^|\s)(?:serve|--serve)(?:\s|$)') -or
        ($_.Name -match '^python(?:w)?\.exe$' -and $command -match '(?:^|\s)-m\s+serve\.server(?:\s|$)' -and
         $command -match $engineArgument)
    })
    if ($roots.Count -eq 0) {
        Write-Host 'Basalt is not running from this folder.'
        exit 0
    }

    $targets = @{}
    foreach ($process in $roots) { $targets[[int]$process.ProcessId] = $process }
    do {
        $added = $false
        foreach ($process in $processes) {
            if ($targets.ContainsKey([int]$process.ParentProcessId) -and
                $process.CreationDate -ge $targets[[int]$process.ParentProcessId].CreationDate -and
                -not $targets.ContainsKey([int]$process.ProcessId)) {
                $targets[[int]$process.ProcessId] = $process
                $added = $true
            }
        }
    } while ($added)

    try {
        $config = Get-Content -LiteralPath (Join-Path $rootPath 'basalt_config.json') -Raw | ConvertFrom-Json
        $port = if ($config.server.port) { [int]$config.server.port } else { 8080 }
        $url = 'http://127.0.0.1:' + $port
        $health = Invoke-RestMethod ($url + '/health') -TimeoutSec 3
        $listener = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            Where-Object { $targets.ContainsKey([int]$_.OwningProcess) })
        if ($health.service -eq 'basalt' -and $listener.Count -gt 0) {
            Write-Host 'Unloading Basalt and saving its expert profile...'
            $result = Invoke-RestMethod ($url + '/unload') -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 65
            if ($result.status -notin @('unloaded', 'not loaded')) {
                Write-Host ('Clean unload unavailable: ' + $result.status)
            }
        }
    } catch {
        Write-Host 'Clean unload unavailable; stopping this Basalt process tree.'
    }

    foreach ($process in $targets.Values) {
        $current = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $process.ProcessId)
        if ($current -and $current.CreationDate -eq $process.CreationDate -and
            $current.ExecutablePath -eq $process.ExecutablePath) {
            Stop-Process -Id $current.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Milliseconds 500
    foreach ($process in $targets.Values) {
        $current = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $process.ProcessId)
        if ($current -and $current.CreationDate -eq $process.CreationDate) {
            throw ('Basalt process ' + $process.ProcessId + ' did not stop.')
        }
    }
    Write-Host 'Basalt stopped.'
} catch {
    Write-Host ('Could not stop Basalt: ' + $_.Exception.Message)
    exit 1
}
