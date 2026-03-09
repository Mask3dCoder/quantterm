#!/usr/bin/env pwsh
# QuantTerm PowerShell Wrapper Script
# This script provides seamless Windows PowerShell experience for QuantTerm CLI
# Usage: .\quantterm.ps1 [command] [options]

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

python -m quantterm @Args
