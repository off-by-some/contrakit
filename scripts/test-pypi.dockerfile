FROM python:3.10-slim

# Install contrakit from TestPyPI
RUN pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ \
                contrakit

# Test basic functionality
RUN python -c "import contrakit; print('✅ Contrakit installed successfully!')"
RUN python -c "import contrakit; print('📦 Version:', contrakit.__version__)"
RUN python -c "import contrakit; print('🔧 Available classes:', [x for x in dir(contrakit) if not x.startswith('_')][:8])"
RUN python -c "from contrakit import Observatory, Space, Behavior; print('✅ Core imports work!')"
RUN python -c "from contrakit import Observatory; obs = Observatory.create(symbols=['Yes', 'No']); Y = obs.concept('Outcome'); print('✅ Observatory creation works!')"
RUN python -c "from contrakit import Space; space = Space.create(A=[0, 1], B=[0, 1]); print('✅ Space creation works!')"
RUN python -c "print('🎉 All tests passed! Contrakit is working correctly.')"
