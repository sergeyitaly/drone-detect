#!/bin/bash
set -e

echo "Stopping containers..."
docker-compose down

echo "Making sure init.sh is executable..."
chmod +x init.sh

echo "Building docker image drone-detector-app..."
sudo docker build -t drone-detector-app ..

echo "Starting containers..."
docker-compose up -d
	
