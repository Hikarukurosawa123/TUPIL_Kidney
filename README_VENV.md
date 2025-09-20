# TUPIL Kidney - Virtual Environment Setup

This document explains how to set up and use the virtual environment for the TUPIL Kidney project.

## 🚀 Quick Start

### Option 1: Use the activation script (Recommended)
```bash
./activate_venv.sh
```

### Option 2: Manual activation
```bash
source venv/bin/activate
```

## 📦 Installed Packages

The virtual environment includes all necessary packages for the TUPIL Kidney project:

### Core Data Science
- **NumPy** 2.3.3 - Numerical computing
- **Pandas** 2.3.2 - Data manipulation and analysis
- **Matplotlib** 3.10.6 - Plotting and visualization
- **SciPy** 1.16.2 - Scientific computing

### Machine Learning & Deep Learning
- **TensorFlow** 2.20.0 - Deep learning framework
- **Scikit-learn** 1.7.2 - Machine learning library
- **Keras** 3.11.3 - High-level neural networks API

### Image Processing
- **Pillow (PIL)** 11.3.0 - Python Imaging Library
- **Scikit-image** 0.25.2 - Image processing algorithms

### Development Tools
- **Jupyter** 1.1.1 - Jupyter notebook environment
- **IPython** 9.5.0 - Interactive Python shell
- **TQDM** 4.67.1 - Progress bars

## 🛠️ Usage

### Starting Jupyter Lab
```bash
# Activate the environment first
source venv/bin/activate

# Start Jupyter Lab
jupyter lab
```

### Running Python scripts
```bash
# Activate the environment first
source venv/bin/activate

# Run your Python script
python your_script.py
```

### Deactivating the environment
```bash
deactivate
```

## 📁 Project Structure

```
TUPIL_Kidney/
├── venv/                          # Virtual environment
├── data/                          # Data directory
│   ├── lanczos_shape_corrected_only_nc_resized_images/
│   └── model_weights/
├── csv/                           # CSV files
├── requirements.txt               # Package requirements
├── requirements_exact.txt         # Exact package versions
├── activate_venv.sh              # Activation script
└── README_VENV.md                # This file
```

## 🔧 Environment Detection

The project includes automatic environment detection that sets appropriate paths based on whether you're running:
- **Google Colab** (with Google Drive mounted)
- **Google Colab** (without Google Drive)
- **Local environment**

This ensures the code works seamlessly across different environments without manual path changes.

## 📋 Requirements Files

- `requirements.txt` - General requirements with version ranges
- `requirements_exact.txt` - Exact versions of all installed packages

## 🐛 Troubleshooting

### If you encounter import errors:
1. Make sure the virtual environment is activated: `source venv/bin/activate`
2. Check if all packages are installed: `pip list`
3. Reinstall packages if needed: `pip install -r requirements.txt`

### If you need to recreate the environment:
```bash
# Remove existing environment
rm -rf venv

# Create new environment
python3 -m venv venv

# Activate and install packages
source venv/bin/activate
pip install -r requirements.txt
```

## 🎯 Next Steps

1. Activate the virtual environment
2. Start Jupyter Lab: `jupyter lab`
3. Open your notebook and run the cells
4. The environment detection will automatically set the correct paths

Happy coding! 🚀
