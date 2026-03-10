<#
Open local SSH tunnel to remote backend (PowerShell version).

Default mapping:
  127.0.0.1:8000 (local) -> 127.0.0.1:8000 (remote)

Examples:
  ./scripts/open_tunnel.ps1 -HostName "your-autodl-host" -UserName "ubuntu"
  ./scripts/open_tunnel.ps1 -Target "ubuntu@your-autodl-host"
  ./scripts/open_tunnel.ps1 -HostName "your-autodl-host" -UserName "ubuntu" -LocalPort 8000 -RemotePort 8000 -SshPort 22
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Target,

    [Parameter(Mandatory = $false)]
    [string]$HostName,

    [Parameter(Mandatory = $false)]
    [string]$UserName,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 65535)]
    [int]$LocalPort = 8000,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 65535)]
    [int]$RemotePort = 8000,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22
)

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Error "ssh command not found. Please install OpenSSH client first."
    exit 1
}

# Resolve target from either -Target or -HostName/-UserName.
if ([string]::IsNullOrWhiteSpace($Target)) {
    if ([string]::IsNullOrWhiteSpace($HostName)) {
        Write-Host "Usage:"
        Write-Host "  ./scripts/open_tunnel.ps1 -Target user@host [-LocalPort 8000] [-RemotePort 8000] [-SshPort 22]"
        Write-Host "  ./scripts/open_tunnel.ps1 -HostName host [-UserName user] [-LocalPort 8000] [-RemotePort 8000] [-SshPort 22]"
        exit 1
    }
    if ([string]::IsNullOrWhiteSpace($UserName)) {
        $Target = $HostName
    } else {
        $Target = "$UserName@$HostName"
    }
} elseif (($Target -notmatch "@") -and (-not [string]::IsNullOrWhiteSpace($UserName))) {
    $Target = "$UserName@$Target"
}

Write-Host "[tunnel] target: $Target"
Write-Host "[tunnel] local : 127.0.0.1:$LocalPort"
Write-Host "[tunnel] remote: 127.0.0.1:$RemotePort"
Write-Host "[tunnel] ssh port: $SshPort"
Write-Host "[tunnel] press Ctrl+C to close"

$sshArgs = @(
    "-p", "$SshPort",
    "-N",
    "-L", "$LocalPort`:127.0.0.1:$RemotePort",
    "$Target"
)

& ssh @sshArgs
