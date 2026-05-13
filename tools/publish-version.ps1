param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    if ($DryRun) {
        Write-Host ("DRY RUN: git " + ($Arguments -join " "))
        return
    }

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-LatestVersionLabel {
    $versionFiles = Get-ChildItem -Directory -Name "a2k_v*" |
        ForEach-Object {
            $versionFile = Join-Path $_ "VERSION"
            if (Test-Path -LiteralPath $versionFile) {
                [PSCustomObject]@{
                    Directory = $_
                    Version = (Get-Content -LiteralPath $versionFile -TotalCount 1).Trim()
                }
            }
        } |
        Where-Object { $_.Version }

    $latest = $versionFiles |
        Sort-Object {
            if ($_.Directory -match "^a2k_v(.+)$") {
                try {
                    [version]$Matches[1]
                }
                catch {
                    [version]"0.0"
                }
            }
            else {
                [version]"0.0"
            }
        } |
        Select-Object -Last 1

    if ($latest) {
        return "v$($latest.Version)"
    }

    return (Get-Date -Format "yyyyMMdd-HHmm")
}

function Test-AllowedUntrackedPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $normalized = $Path -replace "\\", "/"

    if ($normalized -eq "AGENTS.md") {
        return $false
    }

    return $normalized -match "^a2k_v[^/]+/"
}

$insideWorkTree = (& git rev-parse --is-inside-work-tree).Trim()
if ($insideWorkTree -ne "true") {
    throw "Current directory is not a Git work tree."
}

$currentBranch = (& git branch --show-current).Trim()
if (-not $currentBranch) {
    throw "Cannot publish from a detached HEAD."
}

$remote = (& git remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $remote) {
    throw "Remote 'origin' is not configured."
}

Write-Host "Staging tracked project changes..."
Invoke-Git @("add", "-u")

$untrackedFiles = & git ls-files --others --exclude-standard
$allowedUntracked = @($untrackedFiles | Where-Object { Test-AllowedUntrackedPath $_ })
$skippedUntracked = @($untrackedFiles | Where-Object { -not (Test-AllowedUntrackedPath $_) })

foreach ($file in $allowedUntracked) {
    Write-Host "Staging new project file: $file"
    Invoke-Git @("add", "--", $file)
}

foreach ($file in $skippedUntracked) {
    Write-Host "Skipping unrelated untracked file: $file"
}

$stagedChanges = & git diff --cached --name-only
if (-not $stagedChanges) {
    Write-Host "No publishable changes found."
    exit 0
}

$versionLabel = Get-LatestVersionLabel
$commitMessage = "Release $versionLabel"

Write-Host "Committing staged changes: $commitMessage"
Invoke-Git @("commit", "-m", $commitMessage)

$upstream = (& git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null)
if ($LASTEXITCODE -eq 0 -and $upstream) {
    Write-Host "Pushing to configured upstream: $upstream"
    Invoke-Git @("push")
}
else {
    Write-Host "Pushing to origin/$currentBranch and setting upstream..."
    Invoke-Git @("push", "-u", "origin", $currentBranch)
}
