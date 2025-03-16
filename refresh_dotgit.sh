#!/bin/bash

# Exit immediately if a command fails
set -e

# Get the remote URL of the current repo
REMOTE_URL=$(git config --get remote.origin.url)

if [[ -z "$REMOTE_URL" ]]; then
    echo "Error: Could not determine the remote repository URL."
    exit 1
fi

# TEMP_DIR=$(mktemp -d)
TEMP_DIR=".git_temp"

echo "Cloning repository to a temporary location..."
git clone "$REMOTE_URL" "$TEMP_DIR"

echo "Removing current .git directory..."
rm -rf .git

echo "Restoring .git directory from the cloned repository..."
mv "$TEMP_DIR/.git" .git

rm -rf "$TEMP_DIR"

echo "Successfully refreshed .git directory from remote."
