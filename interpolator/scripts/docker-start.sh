#!/bin/bash
# Start the application using Docker Compose
# This script starts both backend and frontend containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Navigate to project directory
cd "$PROJECT_DIR"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: docker-compose.yml not found in $PROJECT_DIR"
    exit 1
fi

# Start services using docker-compose
echo "Starting application services..."
docker-compose up -d --build

echo ""
echo "Services are starting. Backend will be available at http://localhost:8000"
echo "Frontend will be available at http://localhost:3000"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"



