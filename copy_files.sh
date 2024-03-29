#!/bin/bash

# Prompt the user for the remote IP and FCT directory
read -p "Enter the remote IP to copy the data from: " remote_ip
read -p "Enter the FCT directory: " fct_directory

# Copy the weights folder from the remote server
scp -r "$remote_ip:$fct_directory/weights" .

# Copy the datasets/data folder from the remote server
scp -r "$remote_ip:$fct_directory/datasets/data" .

echo "Files copied successfully!"