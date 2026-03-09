# Windows Installation Guide

This guide covers installing and running QuantTerm on Windows systems.

## Prerequisites

### Python Installation
1. Download Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/)
   - **Important**: Don't install from Microsoft Store - use the direct installer
2. During installation, check "Add Python to PATH"
3. Verify installation:
   ```powershell
   python --version
   ```

## Installation Methods

### Method 1: pip install (Recommended)

```powershell
# Install QuantTerm
pip install quantterm

# Verify installation
quantterm --help
```

If `quantterm` command is not found immediately, see Troubleshooting below.

### Method 2: Development Installation

```powershell
# Clone the repository
git clone https://github.com/quantterm/quantterm.git
cd quantterm

# Install in development mode
pip install -e .

# Verify installation
quantterm --help
```

## Usage

Once installed, you can use QuantTerm commands:

```powershell
# Get stock quote
quantterm quote AAPL

# Get historical data
quantterm history SPY --start 1y

# Technical analysis
quantterm tech indicator AAPL rsi --period 14

# Run backtest
quantterm backtest run --symbol SPY --start 2020-01-01
```

## Troubleshooting

### "quantterm is not recognized" Error

If you see `quantterm is not recognized as an internal or external command`:

**Option 1: Use python -m (Recommended)**
```powershell
python -m quantterm quote AAPL
```

**Option 2: Restart Terminal**
Close and reopen your PowerShell/Command Prompt, then try again. This is necessary if you installed QuantTerm recently and haven't restarted your terminal.

**Option 3: Add Python Scripts to PATH Permanently**
If the command still doesn't work after restarting, you need to add Python Scripts to your system PATH:

PowerShell method (recommended for PowerShell users):
```powershell
# Create a PowerShell profile if you don't have one
if (!(Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force }

# Add Python Scripts to your profile
$pythonScripts = python -c "import sys; print(sys.prefix)" + "\Scripts"
Add-Content -Path $PROFILE -Value "`n`n# Add Python Scripts to PATH`nif (`$env:PATH -notlike '*$pythonScripts*') { `$env:PATH += ';$pythonScripts' }"

# Close and reopen PowerShell - the command should now work
quantterm quote AAPL
```

Command Prompt method (alternative):
```cmd
setx PATH "%PATH%;C:\Users\YOUR_USERNAME\AppData\Local\Packages\Python313\Scripts"

# Restart your terminal/VS Code for changes to take effect
```

**Option 4: Use Wrapper Scripts**
```powershell
# From the quantterm directory
.\scripts\quantterm.bat quote AAPL

# Or in PowerShell
.\scripts\quantterm.ps1 quote AAPL
```

### Verify Installation

```powershell
# Check if QuantTerm is installed
python -c "import quantterm; print(quantterm.__version__)"

# Check executable location
where quantterm
```

## Wrapper Scripts

The project includes wrapper scripts in the `scripts/` directory:

- `scripts/quantterm.bat` - For Command Prompt
- `scripts/quantterm.ps1` - For PowerShell

You can copy these to any directory in your PATH for easy access.

## Common Issues

| Issue | Solution |
|-------|----------|
| Command not found | Use `python -m quantterm` or restart terminal |
| Permission denied | Run PowerShell as Administrator |
| Old version | `pip install --upgrade quantterm` |
| Multiple Python versions | Use `python3` or specify full path |

## Getting Help

- GitHub Issues: https://github.com/quantterm/quantterm/issues
- Documentation: https://quantterm.readthedocs.io
