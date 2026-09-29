$ErrorActionPreference = "Stop"
$image = "server2-system:no-dataset-20260926"
$container = "server2-env-validation-20260927"
$work = $PSScriptRoot

& docker image inspect $image | Set-Content -LiteralPath (Join-Path $work "image-inspect.json") -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw "Imported image does not exist." }

$argsList = @(
    "run", "--name", $container,
    "--network", "none", "--cpus", "4", "--memory", "6g", "--memory-swap", "7g",
    "--shm-size", "1g",
    "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONUNBUFFERED=1",
    "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
    "--env", "OMP_NUM_THREADS=4", "--env", "MKL_NUM_THREADS=4",
    "--mount", "type=bind,source=$work,target=/validation",
    $image,
    "/root/hyperseg/.venv-mask2former/bin/python", "-u", "/validation/validate_environment.py"
)
& docker @argsList
$validationExit = $LASTEXITCODE
& docker inspect $container | Set-Content -LiteralPath (Join-Path $work "container-inspect.json") -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw "Cannot inspect validation container." }
$validationExit | Set-Content -LiteralPath (Join-Path $work "exit_code.txt") -Encoding ASCII
if ($validationExit -ne 0) { throw "Container validation failed with exit code $validationExit." }
Write-Host "Docker environment validation passed."
