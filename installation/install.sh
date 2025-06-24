#!/bin/bash
# Corpus CLI Installation Script
# This script sets up everything needed to run Corpus

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check Python version
check_python() {
    print_header "Checking Python Version"
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
        PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    print_header "Setting Up Virtual Environment"
    
    if [ -d "env" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf env
            python3 -m venv env
            print_success "Virtual environment recreated"
        else
            print_success "Using existing virtual environment"
        fi
    else
        python3 -m venv env
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source env/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    # Upgrade pip first
    pip install --upgrade pip wheel setuptools
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Install corpus command
install_corpus() {
    print_header "Installing Corpus Command"
    
    # Install in development mode
    pip install -e .
    
    # Verify installation
    if command -v corpus &> /dev/null; then
        print_success "Corpus command installed successfully"
    else
        print_error "Corpus command installation failed"
        exit 1
    fi
}

# Check for D2
check_d2() {
    print_header "Checking D2 (Diagram Tool)"
    
    if command -v d2 &> /dev/null; then
        D2_VERSION=$(d2 --version 2>&1 | head -n1)
        print_success "D2 found: $D2_VERSION"
    else
        print_warning "D2 not found (optional, needed for diagram generation)"
        echo
        echo "To install D2:"
        echo "  macOS:    brew install d2"
        echo "  Linux:    curl -fsSL https://d2lang.com/install.sh | sh -s --"
        echo "  Windows:  Visit https://d2lang.com/tour/install"
    fi
}

# Setup initial configuration
setup_config() {
    print_header "Initial Configuration"
    
    # Create config directory
    mkdir -p ~/.corpus
    
    echo "Would you like to configure Corpus now? (recommended)"
    read -p "Configure now? (Y/n) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        corpus config setup
    else
        print_warning "You can configure later with: corpus config setup"
    fi
}

# Main installation flow
main() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════╗"
    echo "║      Corpus CLI Installation          ║"
    echo "║   Document Chat & AI Assistant        ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Run installation steps
    check_python
    setup_venv
    install_dependencies
    install_corpus
    check_d2
    
    # Show success message
    echo -e "\n${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Corpus installation complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}\n"
    
    # Show next steps
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Activate the virtual environment:"
    echo "   ${YELLOW}source env/bin/activate${NC}"
    echo
    echo "2. Configure your API key:"
    echo "   ${YELLOW}corpus config setup${NC}"
    echo "   or"
    echo "   ${YELLOW}corpus config set GEMINI_API_KEY \"your-key\"${NC}"
    echo
    echo "3. Start using Corpus:"
    echo "   ${YELLOW}corpus --help${NC}"
    echo "   ${YELLOW}corpus index ~/Documents${NC}"
    echo "   ${YELLOW}corpus chat${NC}"
    echo
    
    # Ask about configuration
    setup_config
}

# Run main function
main