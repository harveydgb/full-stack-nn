#!/bin/bash
# Test the full pipeline: start backend and frontend in docker, verify they work
# This script tests the complete application workflow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Testing Full Pipeline"
echo "=========================================="
echo ""

# Navigate to project directory
cd "$PROJECT_DIR"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: docker-compose.yml not found in $PROJECT_DIR"
    exit 1
fi

# Step 1: Start services
echo "Step 1: Starting services with docker-compose..."
docker-compose up -d --build

# Step 2: Wait for services to be healthy
echo ""
echo "Step 2: Waiting for services to be healthy..."
BACKEND_HEALTHY=false
FRONTEND_HEALTHY=false
MAX_ATTEMPTS=60  # Wait up to 2 minutes
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    # Check backend health
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        BACKEND_HEALTHY=true
    fi
    
    # Check frontend health
    if curl -s -f http://localhost:3000 > /dev/null 2>&1; then
        FRONTEND_HEALTHY=true
    fi
    
    # If both are healthy, break
    if [ "$BACKEND_HEALTHY" = true ] && [ "$FRONTEND_HEALTHY" = true ]; then
        break
    fi
    
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
    echo -n "."
done

echo ""
echo ""

# Step 3: Verify services
echo "Step 3: Verifying services..."
echo ""

BACKEND_STATUS="✗ FAILED"
FRONTEND_STATUS="✗ FAILED"

if [ "$BACKEND_HEALTHY" = true ]; then
    BACKEND_STATUS="✓ HEALTHY"
    echo "$BACKEND_STATUS - Backend: http://localhost:8000"
    echo "  - Health endpoint: http://localhost:8000/health"
    echo "  - API docs: http://localhost:8000/docs"
else
    echo "$BACKEND_STATUS - Backend not responding"
    echo "  Check logs: docker-compose logs backend"
fi

echo ""

if [ "$FRONTEND_HEALTHY" = true ]; then
    FRONTEND_STATUS="✓ HEALTHY"
    echo "$FRONTEND_STATUS - Frontend: http://localhost:3000"
else
    echo "$FRONTEND_STATUS - Frontend not responding"
    echo "  Check logs: docker-compose logs frontend"
fi

echo ""

# Step 4: Test API endpoints (if backend is healthy)
if [ "$BACKEND_HEALTHY" = true ]; then
    echo "Step 4: Testing API endpoints..."
    echo ""
    
    # Test health endpoint
    echo -n "  Testing GET /health... "
    if curl -s -f http://localhost:8000/health | grep -q "healthy" || curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ PASSED"
    else
        echo "✗ FAILED"
    fi
    
    # Test root endpoint
    echo -n "  Testing GET /... "
    if curl -s -f http://localhost:8000/ > /dev/null 2>&1; then
        echo "✓ PASSED"
    else
        echo "✗ FAILED"
    fi
    
    # Test API docs endpoint
    echo -n "  Testing GET /docs... "
    if curl -s -f http://localhost:8000/docs > /dev/null 2>&1; then
        echo "✓ PASSED"
    else
        echo "✗ FAILED"
    fi
    
    echo ""
fi

# Step 5: Summary
echo "=========================================="
echo "Pipeline Test Summary"
echo "=========================================="
echo ""
echo "Backend:  $BACKEND_STATUS"
echo "Frontend: $FRONTEND_STATUS"
echo ""

if [ "$BACKEND_HEALTHY" = true ] && [ "$FRONTEND_HEALTHY" = true ]; then
    echo "✓ Full pipeline test PASSED"
    echo ""
    echo "The application is running and ready for use:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend API: http://localhost:8000"
    echo "  - API Documentation: http://localhost:8000/docs"
    echo ""
    echo "You can now:"
    echo "  1. Open http://localhost:3000 in your browser"
    echo "  2. Upload a dataset (.pkl file)"
    echo "  3. Train a neural network model"
    echo "  4. Make predictions"
    echo ""
    echo "To stop services: docker-compose down"
    exit 0
else
    echo "✗ Full pipeline test FAILED"
    echo ""
    echo "Please check the logs for more information:"
    echo "  docker-compose logs backend"
    echo "  docker-compose logs frontend"
    echo ""
    exit 1
fi

