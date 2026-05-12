param(
    [string]$PythonExe = "py -3.12"
)

Write-Host "Building QiFlow release..."
& $PythonExe -m pip install -r requirements.txt
& $PythonExe -m pip install pyinstaller

$dist = Join-Path $PSScriptRoot "..\dist"
$release = Join-Path $PSScriptRoot "..\release"

& $PythonExe -m pyinstaller --noconsole --onefile --add-data "assets;assets" --add-data "config;config" ..\main.py

if (Test-Path $release) { Remove-Item $release -Recurse -Force }
New-Item -ItemType Directory -Path $release | Out-Null
Copy-Item (Join-Path $dist "main.exe") (Join-Path $release "QiFlow.exe")
Copy-Item ..\config -Destination (Join-Path $release "config") -Recurse
Copy-Item ..\assets -Destination (Join-Path $release "assets") -Recurse
Write-Host "Release ready at $release"
