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

# Create global corpus command
create_global_command() {
    print_header "Global Command Setup"
    
    # Get the current directory (installation path)
    INSTALL_DIR="$(pwd)"
    
    # Check if global corpus command already exists
    if [ -f "/usr/local/bin/corpus" ]; then
        print_warning "Global corpus command already exists at /usr/local/bin/corpus"
        echo
        echo "Would you like to update it with the current installation path?"
        echo -n "Update existing global command? [Y/n]: "
        read -r REPLY
        
        if [[ -z "$REPLY" ]] || [[ "$REPLY" =~ ^[Yy]$ ]]; then
            UPDATE_EXISTING=true
        else
            print_success "Keeping existing global command"
            GLOBAL_COMMAND_CREATED=true
            return
        fi
    else
        echo "Would you like to create a global 'corpus' command?"
        echo "This will allow you to use 'corpus' from anywhere without activating the virtual environment."
        echo
        echo -n "Create global command? (requires sudo) [Y/n]: "
        read -r REPLY
        
        if [[ ! -z "$REPLY" ]] && [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
            print_warning "Skipped global command setup"
            GLOBAL_COMMAND_CREATED=false
            echo
            echo "To use corpus, you'll need to:"
            echo "  1. cd $INSTALL_DIR"
            echo "  2. source env/bin/activate"
            echo "  3. corpus --help"
            return
        fi
        UPDATE_EXISTING=false
    fi
    
    # Create or update the wrapper
    echo "Creating global command..."
    
    # Create wrapper script
    WRAPPER_CONTENT="#!/bin/bash
# Corpus CLI global wrapper
# Auto-generated by install.sh
cd \"$INSTALL_DIR\" && source env/bin/activate && env/bin/corpus \"\$@\""
    
    # Write to temp file first
    echo "$WRAPPER_CONTENT" > /tmp/corpus-global-wrapper
    
    # Try to move to /usr/local/bin with sudo
    if [ "$UPDATE_EXISTING" = true ]; then
        ACTION="Updated"
    else
        ACTION="Created"
    fi
    
    if sudo mv /tmp/corpus-global-wrapper /usr/local/bin/corpus 2>/dev/null; then
        sudo chmod +x /usr/local/bin/corpus
        print_success "$ACTION global corpus command at /usr/local/bin/corpus"
        echo
        echo "You can now use 'corpus' from anywhere!"
        GLOBAL_COMMAND_CREATED=true
    else
        print_error "Failed to create global command (sudo required)"
        rm -f /tmp/corpus-global-wrapper
        GLOBAL_COMMAND_CREATED=false
        
        # Offer alternative
        echo
        echo "Alternative: Add this alias to your ~/.bashrc or ~/.zshrc:"
        echo "  alias corpus='cd $INSTALL_DIR && source env/bin/activate && corpus'"
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
    
    # Create global command
    create_global_command
    
    # Show next steps based on whether global command was created
    if command -v corpus &> /dev/null && [ -f "/usr/local/bin/corpus" ]; then
        echo -e "\n${BLUE}Next steps:${NC}"
        echo "1. Configure your API key:"
        echo "   ${YELLOW}corpus config setup${NC}"
        echo "   or"
        echo "   ${YELLOW}corpus config set GEMINI_API_KEY \"your-key\"${NC}"
        echo
        echo "2. Start using Corpus:"
        echo "   ${YELLOW}corpus --help${NC}"
        echo "   ${YELLOW}corpus index ~/Documents${NC}"
        echo "   ${YELLOW}corpus chat${NC}"
        echo
        
        # Configuration
        echo "Would you like to configure Corpus now?"
        read -p "Configure now? (Y/n) " -n 1 -r
        echo
        
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            corpus config setup
        else
            print_warning "You can configure later with: corpus config setup"
        fi
    else
        echo -e "\n${YELLOW}IMPORTANT: Activate the virtual environment first!${NC}"
        echo -e "${GREEN}Run this command:${NC}"
        echo -e "  ${YELLOW}source env/bin/activate${NC}\n"
        
        echo -e "${BLUE}Then you can use:${NC}"
        echo "1. Configure your API key:"
        echo "   ${YELLOW}corpus config setup${NC}"
        echo
        echo "2. Start using Corpus:"
        echo "   ${YELLOW}corpus --help${NC}"
        echo "   ${YELLOW}corpus index ~/Documents${NC}"
        echo "   ${YELLOW}corpus chat${NC}"
    fi
}

# Run main function
main