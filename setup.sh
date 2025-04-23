echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
# Detect the shell and activate accordingly
if [[ "$SHELL" == */zsh ]]; then
    source venv/bin/activate
elif [[ "$SHELL" == */bash ]]; then
    source venv/bin/activate
else
    echo "Please activate manually: source venv/bin/activate"
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

echo "  Setup complete. Virtual environment is ready."