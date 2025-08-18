#!/bin/bash
# Quick install script cho Ubuntu/Debian với externally-managed-environment

echo "🚀 Quick Setup cho Smart Waste Management (Ubuntu/Debian)"

# Method 1: Virtual Environment (Recommended)
echo ""
echo "🌍 Method 1: Virtual Environment (Khuyên dùng)"
echo "============================================="

# Create virtual environment
python3 -m venv venv_waste
source venv_waste/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install packages
pip install matplotlib numpy folium branca requests

echo "✅ Virtual environment setup complete!"
echo ""

# Method 2: System packages (fallback)
echo "📦 Method 2: System Packages (fallback)"
echo "========================================"

# Update package list
sudo apt-get update

# Install system packages
sudo apt-get install -y python3-tk python3-dev python3-full
sudo apt-get install -y python3-matplotlib python3-numpy || echo "⚠️ Some packages may not be available"

echo ""
echo "🎯 Usage:"
echo ""
echo "Option A - Virtual Environment:"
echo "  source venv_waste/bin/activate"
echo "  cd system/"
echo "  python enhanced_map_gui.py"
echo ""
echo "Option B - System Python:"
echo "  cd system/"
echo "  python3 enhanced_map_gui.py"
echo ""
echo "Option C - Web Interface (always works):"
echo "  cd system/"
echo "  python3 web_map_interface.py"

# Create quick start script
cat > start_waste_management.sh << 'EOF'
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
EOF

chmod +x start_waste_management.sh

echo ""
echo "🎉 Quick setup complete!"
echo "📜 Run './start_waste_management.sh' to start the application"
