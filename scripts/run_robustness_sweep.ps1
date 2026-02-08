param(
  [string]$OutRoot = "outputs/robustness_sweep_2026-02-07",
  [string]$LEVIR = "data/LEVIR-CD",
  [string]$WHU = "data/WHUCD",
  [string]$CkptLEVIR = "outputs/retrain_2026-02-06/levir2whu/best.pt",
  [string]$CkptWHU = "outputs/retrain_2026-02-06/whu2levir/best.pt",
  [ValidateSet("none","flip","d4")][string]$TTA = "none",
  [ValidateSet("quick","full")][string]$EvalMode = "quick",
  [string]$Indices = "3,4",
  [int]$NumWorkers = 0
)

$ErrorActionPreference = "Stop"

$env:ALBUMENTATIONS_DISABLE_VERSION_CHECK = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

function Run-One {
  param(
    [string]$Name,
    [string]$Checkpoint,
    [string]$DataRoot,
    [string]$Corrupt,
    [string]$PairMode,
    [string]$ExtraArgs
  )

  $outDir = Join-Path $OutRoot $Name
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  $base = @(
    "python", "eval_dino_head.py",
    "--checkpoint", $Checkpoint,
    "--data_root", $DataRoot,
    "--out_dir", $outDir,
    "--thr_mode", "fixed",
    "--thr", "0.5",
    "--smooth_k", "3",
    "--use_minarea",
    "--min_area", "256",
    "--ensemble_strategy", "mean_logit",
    "--ensemble_indices", $Indices,
    "--tta", $TTA,
    "--num_workers", "$NumWorkers",
    "--corrupt", $Corrupt,
    "--corrupt_pair", $PairMode,
    "--corrupt_seed", "0"
  )

  if ($EvalMode -eq "full") {
    $base += @("--full_eval", "--window", "256", "--stride", "128")
  } else {
    $base += @("--no_full_eval", "--eval_crop", "256")
  }

  if ($ExtraArgs -and $ExtraArgs.Trim().Length -gt 0) {
    $base += $ExtraArgs.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
  }

  Write-Host "`n[$Name] ckpt=$Checkpoint data=$DataRoot corrupt=$Corrupt pair=$PairMode extra=$ExtraArgs"
  $exe = $base[0]
  $argv = $base[1..($base.Count-1)]
  & $exe @argv
}

# Scenarios: in-domain + cross-domain
$scenarios = @(
  @{ name = "A_levir_in_clean";   ckpt = $CkptLEVIR; data = $LEVIR },
  @{ name = "B_whu_in_clean";     ckpt = $CkptWHU;   data = $WHU   },
  @{ name = "C_levir2whu_clean";  ckpt = $CkptLEVIR; data = $WHU   },
  @{ name = "D_whu2levir_clean";  ckpt = $CkptWHU;   data = $LEVIR }
)

foreach ($s in $scenarios) {
  Run-One -Name $s.name -Checkpoint $s.ckpt -DataRoot $s.data -Corrupt "none" -PairMode "correlated" -ExtraArgs ""
}

# A small sweep (quick sanity). Expand as needed.
$pairs = @("correlated", "uncorrelated")

foreach ($pm in $pairs) {
  foreach ($s in $scenarios) {
    Run-One -Name ($s.name + "_gauss_s003_" + $pm) -Checkpoint $s.ckpt -DataRoot $s.data -Corrupt "gaussian" -PairMode $pm -ExtraArgs "--gaussian_sigma 0.03"
    Run-One -Name ($s.name + "_bc_b04_c06_" + $pm) -Checkpoint $s.ckpt -DataRoot $s.data -Corrupt "bc" -PairMode $pm -ExtraArgs "--bc_brightness 0.40 --bc_contrast 0.60"
    Run-One -Name ($s.name + "_jpeg_q50_" + $pm) -Checkpoint $s.ckpt -DataRoot $s.data -Corrupt "jpeg" -PairMode $pm -ExtraArgs "--jpeg_quality 50"
  }
}

Write-Host "`n[OK] Saved robustness evals under: $OutRoot"
