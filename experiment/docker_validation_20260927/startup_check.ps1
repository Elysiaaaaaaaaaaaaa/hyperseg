$ErrorActionPreference = 'Stop'
$image = 'server2-system:no-dataset-20260926'
docker image inspect $image --format 'WorkDir={{.Config.WorkingDir}} Cmd={{json .Config.Cmd}}'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
'echo DEFAULT_STARTUP_OK; pwd; exit; # Windows CRLF' | docker run --rm -i --network none $image
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Get-Content -Raw (Join-Path $PSScriptRoot 'startup_check.py') | docker run --rm -i --network none --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 $image python -u -
exit $LASTEXITCODE
