<#
Use the existing project environment, or python on PATH for a fresh checkout.
Examples: .\run.ps1 annotate; .\run.ps1 train-event -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Task = 'list',
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TaskArgs = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PythonExe = Join-Path $PSScriptRoot 'tflm_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
$PythonArgs = @('-B', (Join-Path $PSScriptRoot 'run.py'))
if ($DryRun) { $PythonArgs += '--dry-run' }
$PythonArgs += $Task
$PythonArgs += $TaskArgs

& $PythonExe @PythonArgs
exit $LASTEXITCODE
