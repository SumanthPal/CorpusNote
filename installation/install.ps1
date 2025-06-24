# Corpus CLI Installation Script for Windows
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

# Colors and formatting
function Write-Header {
    param($Text)
    Write-Host "`n==== $Text ====" -ForegroundColor Blue
}

function Write-Success {
    param($Text)
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Write-Error {
    param($Text)
    Write-Host "✗ $Text" -ForegroundColor Red
}

function Write-Warning {
    param($Text)
    Write-Host "⚠ $Text" -ForegroundColor Yellow
}

# Check Python version
function Check-Python {
    Write-Header "Checking Python Version"
    
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            
            if ($major -eq 3 -and $minor -ge 8) {
                Write-Success "Python $major.$minor found"
            } else {
                Write-Error "Python 3.8+ required, found $major.$minor"
                exit 1
            }
        }
    } catch {
        Write-Error "Python not found. Please install Python 3.8 or higher from python.org"
        exit 1
    }
}

# Create virtual environment
function Setup-Venv {
    Write-Header "Setting Up Virtual Environment"
    
    if (Test-Path "env") {
        Write-Warning "Virtual environment already exists"
        $response = Read-Host "Do you want to recreate it? (y/N)"
        
        if ($response -eq 'y') {
            Remove-Item -Recurse -Force env
            python -m venv env
            Write-Success "Virtual environment recreated"
        } else {
            Write-Success "Using existing virtual environment"
        }
    } else {
        python -m venv env
        Write-Success "Virtual environment created"
    }
    
    # Activate virtual environment
    & .\env\Scripts\Activate.ps1
    Write-Success "Virtual environment activated"
}

# Install dependencies
function Install-Dependencies {
    Write-Header "Installing Dependencies"
    
    # Upgrade pip
    python -m pip install --upgrade pip wheel setuptools
    
    # Install requirements
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
        Write-Success "Dependencies installed"
    } else {
        Write-Error "requirements.txt not found"
        exit 1
    }
}

# Install corpus command
function Install-Corpus {
    Write-Header "Installing Corpus Command"
    
    # Install in development mode
    pip install -e .
    
    # Verify installation
    try {
        $corpusPath = (Get-Command corpus -ErrorAction SilentlyContinue).Path
        if ($corpusPath) {
            Write-Success "Corpus command installed successfully"
        }
    } catch {
        Write-Error "Corpus command installation failed"
        exit 1
    }
}

# Check for D2
function Check-D2 {
    Write-Header "Checking D2 (Diagram Tool)"
    
    try {
        $d2Version = d2 --version 2>&1
        Write-Success "D2 found: $d2Version"
    } catch {
        Write-Warning "D2 not found (optional, needed for diagram generation)"
        Write-Host ""
        Write-Host "To install D2:"
        Write-Host "  Visit: https://d2lang.com/tour/install"
        Write-Host "  Or use: winget install terrastruct.d2"
    }
}

# Main installation
function Main {
    Write-Host "`n" -NoNewline
    Write-Host "╔═══════════════════════════════════════╗" -ForegroundColor Blue
    Write-Host "║      Corpus CLI Installation          ║" -ForegroundColor Blue
    Write-Host "║   Document Chat & AI Assistant        ║" -ForegroundColor Blue
    Write-Host "╚═══════════════════════════════════════╝" -ForegroundColor Blue
    Write-Host ""
    
    # Run installation steps
    Check-Python
    Setup-Venv
    Install-Dependencies
    Install-Corpus
    Check-D2
    
    # Show success message
    Write-Host "`n════════════════════════════════════════" -ForegroundColor Green
    Write-Success "Corpus installation complete!"
    Write-Host "════════════════════════════════════════`n" -ForegroundColor Green
    
    # Show next steps
    Write-Host "Next steps:" -ForegroundColor Blue
    Write-Host "1. The virtual environment is already activated"
    Write-Host ""
    Write-Host "2. Configure your API key:" -ForegroundColor Yellow
    Write-Host "   corpus config setup" -ForegroundColor Yellow
    Write-Host "   or"
    Write-Host "   corpus config set GEMINI_API_KEY `"your-key`"" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "3. Start using Corpus:" -ForegroundColor Yellow
    Write-Host "   corpus --help" -ForegroundColor Yellow
    Write-Host "   corpus index ~\Documents" -ForegroundColor Yellow
    Write-Host "   corpus chat" -ForegroundColor Yellow
    Write-Host ""
    
    # Ask about configuration
    $response = Read-Host "Would you like to configure Corpus now? (Y/n)"
    if ($response -ne 'n') {
        corpus config setup
    } else {
        Write-Warning "You can configure later with: corpus config setup"
    }
}

# Run main function
Main