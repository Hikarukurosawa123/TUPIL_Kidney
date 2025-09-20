#!/bin/bash
# Activation script for TUPIL Kidney virtual environment

echo "🚀 Activating TUPIL Kidney virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📍 Current Python: $(which python)"
echo "🐍 Python version: $(python --version)"
echo ""
echo "📦 Installed packages:"
echo "- NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "- Pandas: $(python -c 'import pandas; print(pandas.__version__)')"
echo "- TensorFlow: $(python -c 'import tensorflow; print(tensorflow.__version__)')"
echo "- Scikit-learn: $(python -c 'import sklearn; print(sklearn.__version__)')"
echo "- Scikit-image: $(python -c 'import skimage; print(skimage.__version__)')"
echo ""
echo "🎯 Ready to run your TUPIL Kidney project!"
echo "💡 To start Jupyter: jupyter lab"
echo "💡 To deactivate: deactivate"
