#!/bin/bash

# Prompt the user for the remote IP and FCT directory
read -p "Enter the remote IP to copy the data to: " remote_ip
read -p "Enter the FCT directory: " fct_directory

# Copy the weights folder to the remote server
scp -r weights "$remote_ip:$fct_directory"

# Copy the datasets/data folder to the remote server
scp -r datasets/data "$remote_ip:$fct_directory"

echo "Files copied successfully!"