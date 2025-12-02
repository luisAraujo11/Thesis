#!/bin/bash

# Define the directories to be removed
directories=("logs" "outputs" "uploads")

# Loop through each directory
for dir in "${directories[@]}"; do
  if [ -d "$dir" ]; then
    echo "Removing directory: $dir"
    rm -rf "$dir"
  else
    echo "Directory does not exist: $dir"
  fi
done

# Confirmation message
echo "Specified directories have been processed."
