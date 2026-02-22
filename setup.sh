#!/usr/bin/env bash
# Setup script — creates venv and installs all dependencies
set -e

echo "🔧 Setting up Quant project..."

# Create venv if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Created .venv"
fi

# Install dependencies
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo "   Activate with: source .venv/bin/activate"
echo "   Run backtests: python run_backtests.py"
echo "   Fetch data:    python scripts/data_pipeline.py"
