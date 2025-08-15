#!/bin/bash
echo "🗑️ Starting Smart Waste Management System..."

if [ -d "venv_waste" ]; then
    echo "🌍 Activating virtual environment..."
    source venv_waste/bin/activate
fi

cd system/

echo ""
echo "Choose interface:"
echo "1. 🖥️  Desktop GUI (enhanced_map_gui.py)"  
echo "2. 🌐 Web Interface (web_map_interface.py)"
echo ""

read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo "Starting Desktop GUI..."
        python enhanced_map_gui.py
        ;;
    2)
        echo "Starting Web Interface..."
        python web_map_interface.py
        ;;
    *)
        echo "Invalid choice. Starting Web Interface..."
        python web_map_interface.py
        ;;
esac
