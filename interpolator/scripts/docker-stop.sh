#!/bin/bash
# Stop the application using Docker Compose
# This script stops both backend and frontend containers

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

# Stop services using docker-compose
echo "Stopping application services..."
docker-compose down

echo ""
echo "Services stopped."

