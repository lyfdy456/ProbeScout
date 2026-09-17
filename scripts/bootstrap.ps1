param([ValidateSet('basic', 'full')][string]$Edition = 'basic', [switch]$NoBrowser, [switch]$Configure)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$repo = Split-Path $PSScriptRoot -Parent
$local = Join-Path $repo '.probescout'
try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'This launcher requires Windows x64. See docs/manual_setup.md for other platforms.'
    }
    New-Item -ItemType Directory -Force -Path $local | Out-Null
    $pins = Get-Content -LiteralPath (Join-Path $repo 'manifests/launcher.json') -Raw | ConvertFrom-Json
    $uv = Join-Path $local 'uv/uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        Write-Host 'Preparing the local Python runtime (first launch only)...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $archive = Join-Path $local 'uv.zip'
        Invoke-WebRequest -UseBasicParsing -Uri $pins.uv.url -OutFile $archive
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pins.uv.sha256) {
            throw 'uv download checksum mismatch. Run the launcher again to retry.'
        }
        Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $local 'uv') -Force
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $local 'python'
    $env:UV_CACHE_DIR = Join-Path $local 'uv-cache'
    $env:UV_NO_PROGRESS = '1'
    $env:PYTHONUTF8 = '1'
    $env:PROBESCOUT_UV = $uv
    & $uv python install $pins.python --no-bin
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed. Check your connection and retry.' }
    $python = & $uv python find --managed-python $pins.python
    if ($LASTEXITCODE -ne 0) { throw 'Cannot locate the downloaded Python runtime.' }
    $launcherArgs = @((Join-Path $PSScriptRoot 'launch.py'), '--edition', $Edition)
    if ($NoBrowser) { $launcherArgs += '--no-browser' }
    if ($Configure) { $launcherArgs += '--configure' }
    & $python @launcherArgs
    if ($LASTEXITCODE -ne 0) { throw 'Launcher stopped. See .probescout/launcher.log for details.' }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
