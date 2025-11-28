<#
PowerShell helper to automate SSH key generation, SSH-agent setup, and pushing to GitHub.
Usage:
  .\git_setup_and_push.ps1 -Method SSH   # Automates SSH key generation and pushes via SSH
  .\git_setup_and_push.ps1 -Method PAT   # Use a Personal Access Token for HTTPS push

Security Notes:
- This script does not store your PAT. If you use the PAT method, it will temporarily set the remote with the token and reset it afterwards.
- Always treat private keys and tokens as secrets.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("SSH","PAT")]
    [string]$Method = "SSH",
    [Parameter(Mandatory=$false)]
    [string]$Pat = $null
)

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

Push-Location (Get-Location)

# Get remote URL and repo info
$origUrl = git remote get-url origin 2>$null
if (-not $origUrl) { Write-Host "No remote origin found; add your repo remote first."; exit 1 }
Write-Host "Current origin: $origUrl"

$gitRoot = (git rev-parse --show-toplevel) -replace '\\','/'
$repoName = git rev-parse --show-toplevel | Split-Path -Leaf

if ($Method -eq "SSH") {
    $pubKeyPath = "$env:USERPROFILE\.ssh\id_ed25519.pub"
    $privKeyPath = "$env:USERPROFILE\.ssh\id_ed25519"

    if (-not (Test-Path -Path $pubKeyPath)) {
        Write-Host "No ed25519 public key found. Creating a new ed25519 key..."
        ssh-keygen -t ed25519 -f $privKeyPath -C "`$(git config user.email)" -N "" | Out-Null
        if (-not (Test-Path -Path $pubKeyPath)) { Fail "Key creation failed." }
        Write-Host "Created new key: $privKeyPath"
    } else {
        Write-Host "Public key found at: $pubKeyPath"
    }

    # Start ssh-agent
    Write-Host "Starting ssh-agent service..."
    try { Start-Service -Name ssh-agent -ErrorAction Stop } catch { }
    try { Set-Service -Name ssh-agent -StartupType Automatic -ErrorAction Stop } catch { }

    # Add key to agent
    ssh-add $privKeyPath

    # Copy public key to clipboard
    $pub = Get-Content $pubKeyPath -Raw
    try { Set-Clipboard $pub } catch { Write-Host "Couldn't set clipboard - copy the key manually: `n$pub" }

    Write-Host "Your public key has been copied to the clipboard. Open GitHub -> Settings -> SSH and GPG keys, add a new SSH key and paste it, then press Enter to continue..."
    Start-Process https://github.com/settings/keys
    Read-Host "Press Enter after adding the SSH key in GitHub"

    Write-Host "Switching remote to SSH url..."
    # Build SSH URL from current remote (if it's https) or re-use existing
    $sshUrl = $origUrl
    if ($sshUrl -match 'https://github.com/(.+?)/(.+?)(\.git)?$') {
        $owner = $Matches[1]
        $repo = $Matches[2]
        $sshUrl = "git@github.com:$owner/$repo.git"
    }
    git remote set-url origin $sshUrl
    Write-Host "Remote set to: $(git remote get-url origin)"

    Write-Host "Testing SSH connection..."
    ssh -T git@github.com

    Write-Host "Attempting to push to origin/main..."
    git push -u origin main

} elseif ($Method -eq "PAT") {
    if (-not $Pat) {
        $Pat = Read-Host "Enter a PAT with 'repo' scope (input will be hidden)" -AsSecureString
        $Pat = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($Pat))
    }

    # Build an HTTPS remote with token temporarily
    $orig = git remote get-url origin
    if ($orig -notmatch 'https://') {
        Write-Host "Your current origin doesn't appear to be HTTPS: $orig" -ForegroundColor Yellow
        Write-Host "Attempting to set an HTTPS remote temporarily..."
        if ($orig -match 'git@github.com:(.+?)/(.+?)(\.git)?$') {
            $owner = $Matches[1]
            $repo = $Matches[2]
        } else {
            Fail "Can't extract owner/repo from existing remote: $orig" }
        $httpsRemote = "https://github.com/$owner/$repo.git"
    } else { $httpsRemote = $orig }

    # Temporarily set remote with PAT
    $tempRemote = $httpsRemote -replace 'https://', "https://$Pat@"
    git remote set-url origin $tempRemote

    Write-Host "Pushing to origin using PAT..."
    git push -u origin main

    # Reset origin to standard URL (no token)
    git remote set-url origin $httpsRemote
}

Pop-Location
Write-Host "Done." -ForegroundColor Green
