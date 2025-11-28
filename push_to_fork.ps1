<#
If you don't have push access, use this script to push to your fork.
Usage: run after you fork the repo on GitHub
  .\push_to_fork.ps1 -ForkUrl "git@github.com:yourusername/OIBSIP.git"
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$ForkUrl
)

# Validate
if (-not $ForkUrl) { Write-Host "ForkUrl required"; exit 1 }

# Add or update remote 'myfork'
$existing = git remote get-url myfork 2>$null
if ($existing) { Write-Host "myfork remote already exists: $existing" }
else { git remote add myfork $ForkUrl }

Write-Host "Pushing to myfork..."
git push -u myfork main
Write-Host "Done; open GitHub and create a PR from your fork to upstream."