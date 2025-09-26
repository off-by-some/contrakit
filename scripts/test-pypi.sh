#!/bin/bash
#
# TestPyPI Package Verification Script
# Tests contrakit installation and functionality from TestPyPI using Docker
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🐳 Contrakit TestPyPI Verification"
echo "=================================="

# Build and test the image
echo "📦 Building Docker test image..."
if ! docker build -f "$SCRIPT_DIR/test-pypi.dockerfile" -t contrakit-test . > /dev/null 2>&1; then
    echo "❌ Docker build failed"
    echo ""
    echo "Possible issues:"
    echo "  • TestPyPI package not available or corrupted"
    echo "  • Network connectivity issues"
    echo "  • Docker not running"
    echo ""
    echo "Check: https://test.pypi.org/project/contrakit/"
    exit 1
fi

echo "✅ Docker build successful"
echo "🎉 Contrakit installs correctly from TestPyPI"

# Run functional tests
echo ""
echo "🧪 Running functionality tests..."

TEST_CMD="python -c \"
import contrakit
from contrakit import Observatory

# Test the quickstart example
obs = Observatory.create(symbols=['Yes','No'])
Y = obs.concept('Outcome')
with obs.lens('ExpertA') as A: A.perspectives[Y] = {'Yes': 0.8, 'No': 0.2}
with obs.lens('ExpertB') as B: B.perspectives[Y] = {'Yes': 0.3, 'No': 0.7}

behavior = (A | B).to_behavior()
print('✅ Quickstart example works')
print('📊 alpha* =', round(behavior.alpha_star, 3))
print('📊 K(P) =', round(behavior.contradiction_bits, 3), 'bits')
\""

if docker run --rm contrakit-test bash -c "$TEST_CMD" > /dev/null 2>&1; then
    echo "✅ All functionality tests passed"
    echo ""
    echo "📦 Package Status: VERIFIED"
    echo "   • Installation: ✅ Working"
    echo "   • Imports: ✅ Working"
    echo "   • Core functionality: ✅ Working"
    echo "   • Quickstart example: ✅ Working"
else
    echo "❌ Functionality tests failed"
    echo ""
    echo "📦 Package Status: ISSUES DETECTED"
    echo "   • Check TestPyPI package contents"
    echo "   • Verify dependencies are available"
    exit 1
fi

# Clean up
echo ""
echo "🧹 Cleaning up..."
docker rmi contrakit-test > /dev/null 2>&1
echo "✅ Cleanup complete"

echo ""
echo "🎉 TestPyPI verification complete!"
echo "🚀 Ready for production deployment"
