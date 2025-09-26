#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🐳 Testing Contrakit from TestPyPI using Docker"
echo "=============================================="

# Build and test the image
echo "📦 Building Docker test image..."
docker build -f "$SCRIPT_DIR/test-pypi.dockerfile" -t contrakit-test .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker build successful!"
    echo "🎉 Contrakit installs and works correctly from TestPyPI!"
    
    # Run a quick additional test
    echo ""
    echo "🧪 Running additional functionality test..."
    docker run --rm contrakit-test python -c "
import contrakit
from contrakit import Observatory

# Test the quickstart example
obs = Observatory.create(symbols=['Yes','No'])
Y = obs.concept('Outcome')
with obs.lens('ExpertA') as A: A.perspectives[Y] = {'Yes': 0.8, 'No': 0.2}
with obs.lens('ExpertB') as B: B.perspectives[Y] = {'Yes': 0.3, 'No': 0.7}

behavior = (A | B).to_behavior()
print('✅ Quickstart example works!')
print('📊 alpha*:', round(behavior.alpha_star, 3))
print('📊 K(P):  ', round(behavior.contradiction_bits, 3), 'bits')
"
    
    if [ $? -eq 0 ]; then
        echo "✅ All tests passed!"
    else
        echo "❌ Additional tests failed"
    fi
    
    # Clean up automatically
    echo ""
    echo "🧹 Cleaning up Docker image..."
    docker rmi contrakit-test > /dev/null 2>&1
    echo "✅ Cleanup complete!"
    
else
    echo "❌ Docker build failed - there's an issue with the TestPyPI package"
    exit 1
fi

echo ""
echo "🎉 TestPyPI package verification complete!"
