# Exit immediately if a command fails
set -e

# Create a Python virtual environment called 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install packages from requirements.txt
pip install -r requirements.txt

echo "Setup complete! Virtual environment created and packages installed."