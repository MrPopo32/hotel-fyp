param(
	[string]$RunId = (Get-Date -Format "yyyyMMdd_HHmmss"),
	[int]$BatchSize = 10,
	[Nullable[int]]$MaxHotels = $null,
	[Nullable[int]]$MaxUsers = $null,
	[Nullable[int]]$MaxTargets = $null
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "logs"
$ResultsDir = Join-Path $ProjectRoot "results"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$SummaryLog = Join-Path $LogDir ("overnight_pipeline_" + $RunId + ".log")
$ResultsPath = Join-Path $ResultsDir ("eval_results_" + $RunId + ".txt")

function Write-Step {
	param([string]$Message)
	$Line = "[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] " + $Message
	$Line | Tee-Object -FilePath $SummaryLog -Append
}

function Run-Step {
	param(
		[string]$Name,
		[string]$LogName,
		[string[]]$CommandArgs
	)

	$StepLog = Join-Path $LogDir $LogName
	Write-Step ("Starting: " + $Name)
	& python @CommandArgs *>&1 | Tee-Object -FilePath $StepLog
	if ($LASTEXITCODE -ne 0) {
		throw ($Name + " failed with exit code " + $LASTEXITCODE)
	}
	Write-Step ("Finished: " + $Name)
}

if (-not $env:OPENAI_API_KEY) {
	throw "OPENAI_API_KEY is not set."
}

Write-Step ("Overnight pipeline run id: " + $RunId)
Write-Step "Using API key from environment. The key is not written to logs."

$BuildVectorArgs = @(
	"scripts\build_vectors.py",
	"--batch-size", [string]$BatchSize
)

Run-Step `
	"Run API sentiment enrichment" `
	("build_vectors_" + $RunId + ".log") `
	$BuildVectorArgs

Run-Step `
	"Build hotel profile dataset" `
	("build_dataset_" + $RunId + ".log") `
	@("scripts\build_dataset.py")

"Evaluation Results - " + $RunId | Out-File -FilePath $ResultsPath -Encoding utf8
"" | Out-File -FilePath $ResultsPath -Append -Encoding utf8
$NonPersonalizedArgs = @("scripts\run_eval.py", "--task", "non_personalized")
if ($null -ne $MaxHotels) {
	$NonPersonalizedArgs += @("--max-hotels", [string]$MaxHotels)
}
"Command 1: python " + ($NonPersonalizedArgs -join " ") | Out-File -FilePath $ResultsPath -Append -Encoding utf8
& python @NonPersonalizedArgs *>&1 |
	Tee-Object -FilePath (Join-Path $LogDir ("eval_non_personalized_" + $RunId + ".log")) |
	Out-File -FilePath $ResultsPath -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) {
	throw "Non-personalized evaluation failed with exit code $LASTEXITCODE"
}

"" | Out-File -FilePath $ResultsPath -Append -Encoding utf8
$PersonalizedArgs = @("scripts\run_eval.py", "--task", "personalized")
if ($null -ne $MaxUsers) {
	$PersonalizedArgs += @("--max-users", [string]$MaxUsers)
}
if ($null -ne $MaxTargets) {
	$PersonalizedArgs += @("--max-targets", [string]$MaxTargets)
}
"Command 2: python " + ($PersonalizedArgs -join " ") | Out-File -FilePath $ResultsPath -Append -Encoding utf8
& python @PersonalizedArgs *>&1 |
	Tee-Object -FilePath (Join-Path $LogDir ("eval_personalized_" + $RunId + ".log")) |
	Out-File -FilePath $ResultsPath -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) {
	throw "Personalized evaluation failed with exit code $LASTEXITCODE"
}

Write-Step ("Results saved to: " + $ResultsPath)
Remove-Item Env:\OPENAI_API_KEY -ErrorAction SilentlyContinue
