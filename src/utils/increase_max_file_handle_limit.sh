#!/bin/bash

if [ "$EUID" -ne 0 ]
  then echo "Please run this script as root."
  exit
fi

LIMITS_FILE="/etc/security/limits.conf"
MAX_OPEN_FILES=65536
USERNAME=$(whoami)

echo "Setting open files limit for user $USERNAME to $MAX_OPEN_FILES..."

echo "$USERNAME soft nofile $MAX_OPEN_FILES" | sudo tee -a $LIMITS_FILE
echo "$USERNAME hard nofile $MAX_OPEN_FILES" | sudo tee -a $LIMITS_FILE

echo "Done. Please reboot your system for the changes to take effect."
