param(
    [Parameter(Mandatory = $true)]
    [string]$Root
)

$ErrorActionPreference = 'Stop'

$pgCtl = Join-Path $Root 'workspace\pgsql\pgsql\bin\pg_ctl.exe'
$dataDir = Join-Path $Root 'workspace\pgdata'
$logFile = Join-Path $Root 'workspace\pgvector.log'
$port = 5433

$arguments = "-D `"$dataDir`" -l `"$logFile`" " +
    "-o `"-p $port -c listen_addresses=127.0.0.1`" start"

Start-Process -FilePath $pgCtl -ArgumentList $arguments -WindowStyle Hidden
