#!/bin/bash
# Start the PKM server
# Usage: ./start.sh [port]

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$APP_DIR/.venv/bin"
PORT="${1:-8090}"

echo "PKM Server"
echo "   Dir:  $APP_DIR"
echo "   Port: $PORT"
echo ""

# Check venv
if [ ! -f "$VENV/uvicorn" ]; then
    echo "Error: venv not found. Run:"
    echo "   python3 -m venv $APP_DIR/.venv"
    echo "   $VENV/pip install -r $APP_DIR/scripts/requirements.txt"
    exit 1
fi

# Check .env
if [ ! -f "$APP_DIR/.env" ]; then
    echo "Error: .env not found at $APP_DIR/.env"
    exit 1
fi

# Ensure data directory exists
mkdir -p "$APP_DIR/data"

cd "$APP_DIR"
exec "$VENV/uvicorn" scripts.server:app --host 0.0.0.0 --port "$PORT"
