#!/bin/bash

# Setup script for GitHub repository
# Run this script from the Applied-ML-HW directory

echo "Setting up GitHub repository..."

# Remove existing .git if it exists and has issues
if [ -d ".git" ]; then
    echo "Removing existing .git directory..."
    rm -rf .git
fi

# Initialize git repository
echo "Initializing git repository..."
git init --initial-branch=main

# Add all files
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Applied ML HW1 with notes and Gradio app"

echo ""
echo "✅ Git repository initialized!"
echo ""
echo "Next steps:"
echo "1. Go to https://github.com/new and create a new repository"
echo "2. Name it something like 'Applied-ML-HW' (don't initialize with README)"
echo "3. Copy the repository URL"
echo "4. Run these commands:"
echo "   git remote add origin <YOUR_REPO_URL>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Then connect it to Hugging Face Spaces!"
