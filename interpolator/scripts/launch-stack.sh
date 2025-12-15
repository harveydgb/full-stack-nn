#!/bin/bash
# Launch the entire technology stack locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "5D Neural Network Interpolator"
echo "Local Stack Launcher"
echo "=========================================="
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    echo "Install from: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    echo "Please start Docker and try again"
    exit 1
fi

# Navigate to project directory
cd "$PROJECT_DIR"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: docker-compose.yml not found in $PROJECT_DIR"
    exit 1
fi

# Start services
echo "Building and starting services..."
echo ""

docker-compose up -d --build

# Wait for services to initialize
echo ""
echo "Waiting for services to be healthy..."
echo ""

BACKEND_HEALTHY=false
FRONTEND_HEALTHY=false
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    # Check backend health
    if docker-compose ps backend 2>/dev/null | grep -q "healthy"; then
        BACKEND_HEALTHY=true
    fi
    
    # Check frontend health
    if docker-compose ps frontend 2>/dev/null | grep -q "healthy"; then
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

# Display results
echo "=========================================="
echo "Services Status"
echo "=========================================="
echo ""

if [ "$BACKEND_HEALTHY" = true ]; then
    echo "✓ Backend: http://localhost:8000"
    echo "  - API Docs (Swagger): http://localhost:8000/docs"
    echo "  - API Docs (ReDoc): http://localhost:8000/redoc"
    echo "  - Health Check: http://localhost:8000/health"
else
    echo "✗ Backend: Starting... (may take a moment)"
    echo "  Check logs: docker-compose logs backend"
fi

echo ""

if [ "$FRONTEND_HEALTHY" = true ]; then
    echo "✓ Frontend: http://localhost:3000"
else
    echo "✗ Frontend: Starting... (may take a moment)"
    echo "  Check logs: docker-compose logs frontend"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Open http://localhost:3000 in your browser"
echo ""
echo "2. Upload your dataset (.pkl file) using the web interface"
echo "   - The dataset should have exactly 5 features"
echo "   - Format: {'X': numpy array (n_samples, 5), 'y': numpy array (n_samples,)}"
echo ""
echo "3. Configure training parameters and train the model"
echo "   - Adjust hidden layer sizes, learning rate, etc."
echo "   - Click 'Train model' to start training"
echo ""
echo "4. Make predictions with your trained model"
echo "   - Enter 5 feature values"
echo "   - Click 'Predict' to get predictions"
echo ""
echo "=========================================="
echo "Useful Commands"
echo "=========================================="
echo ""
echo "Stop services:      docker-compose down"
echo "View logs:          docker-compose logs -f"
echo "View backend logs:  docker-compose logs -f backend"
echo "View frontend logs: docker-compose logs -f frontend"
echo "Restart services:   docker-compose restart"
echo ""
echo "=========================================="

