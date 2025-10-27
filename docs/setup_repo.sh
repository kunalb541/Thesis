#!/bin/bash
# Automated setup script for thesis-microlens repository
# Run this on your laptop to create the complete directory structure

echo "=========================================="
echo "Thesis Microlensing Setup Script"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "setup_repo.sh" ]; then
    echo "Error: Please run this script from the thesis-microlens directory"
    exit 1
fi

echo "Creating directory structure..."

# Create main directories
mkdir -p code
mkdir -p slurm
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p results
mkdir -p logs

# Create placeholder files for empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch results/.gitkeep
touch logs/.gitkeep

echo "✓ Directory structure created"

# Make shell scripts executable
if [ -d "slurm" ]; then
    chmod +x slurm/*.sh 2>/dev/null
    echo "✓ Shell scripts made executable"
fi

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git branch -M main
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi

# Check if .gitignore exists
if [ ! -f ".gitignore" ]; then
    echo "Warning: .gitignore not found. Please create it!"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review all files in code/ and slurm/"
echo "2. Commit changes: git add . && git commit -m 'Initial commit'"
echo "3. Create GitHub repo at https://github.com/new"
echo "4. Link remote: git remote add origin https://github.com/USERNAME/thesis-microlens.git"
echo "5. Push: git push -u origin main"
echo ""
echo "Then on the cluster:"
echo "1. git clone https://github.com/USERNAME/thesis-microlens.git"
echo "2. Follow SETUP_GUIDE.md for installation"
echo ""
