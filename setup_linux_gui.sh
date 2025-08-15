#!/bin/bash
# Script cài đặt GUI cho Linux - Giải quyết vấn đề GUI trên Linux

echo "🐧 Thiết lập GUI cho Smart Waste Management trên Linux..."

# Kiểm tra display
if [ -z "$DISPLAY" ]; then
    echo "⚠️ Không phát hiện DISPLAY environment variable"
    echo "💡 Gợi ý:"
    echo "   - Nếu dùng SSH: ssh -X username@server"
    echo "   - Nếu dùng WSL: cài đặt VcXsrv hoặc X410"
    echo "   - Sử dụng web interface thay thế"
else
    echo "✅ DISPLAY detected: $DISPLAY"
fi

# Kiểm tra Python version và externally-managed-environment
echo "🐍 Kiểm tra Python environment..."
PYTHON_VERSION=$(python3 --version)
echo "Python version: $PYTHON_VERSION"

# Detect Linux distribution
if command -v apt-get &> /dev/null; then
    echo "📍 Phát hiện Ubuntu/Debian system"
    
    echo "Updating package list..."
    sudo apt-get update
    
    echo "Installing system dependencies..."
    sudo apt-get install -y python3-tk python3-dev python3-venv python3-full
    sudo apt-get install -y libgl1-mesa-glx  # For matplotlib
    
    # Try to install system packages first
    echo "Trying system packages..."
    sudo apt-get install -y python3-matplotlib python3-numpy || echo "⚠️ System packages not available"
    
elif command -v yum &> /dev/null; then
    echo "📍 Phát hiện CentOS/RHEL system"
    sudo yum install -y tkinter python3-devel python3-venv
    
elif command -v dnf &> /dev/null; then
    echo "📍 Phát hiện Fedora system"  
    sudo dnf install -y python3-tkinter python3-devel python3-venv
    
else
    echo "❌ Không nhận diện được Linux distribution"
    echo "💡 Vui lòng cài đặt thủ công:"
    echo "   - python3-tk"
    echo "   - python3-dev"
    echo "   - python3-venv"
fi

# Setup virtual environment
echo "🌍 Thiết lập Virtual Environment..."
VENV_DIR="venv_waste_management"

if [ ! -d "$VENV_DIR" ]; then
    echo "Tạo virtual environment..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment đã tồn tại"
fi

echo "Kích hoạt virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip trong venv
echo "Upgrading pip trong virtual environment..."
pip install --upgrade pip

# Install Python packages trong venv
echo "🐍 Cài đặt Python packages trong virtual environment..."

# Basic packages
echo "Installing basic packages..."
pip install matplotlib numpy

# Enhanced UI packages  
echo "Installing web mapping packages..."
pip install folium branca

# Try to install PyQt5 as alternative
echo "🎨 Thử cài đặt PyQt5 (GUI alternative)..."
pip install PyQt5 || echo "⚠️ PyQt5 installation failed - tkinter sẽ được dùng thay thế"

# Additional useful packages
echo "Installing additional packages..."
pip install requests geopy

echo "🧪 Test setup trong virtual environment..."
python3 -c "
import os
print(f'DISPLAY: {os.environ.get(\"DISPLAY\", \"Not set\")}')

try:
    import tkinter
    print('✅ tkinter: Available')
except ImportError:
    print('❌ tkinter: Not available')

try:
    import PyQt5
    print('✅ PyQt5: Available')
except ImportError:
    print('⚠️ PyQt5: Not available')

try:
    import matplotlib
    matplotlib.use('Agg')  # Test non-interactive backend
    print('✅ matplotlib: Available')
except ImportError:
    print('❌ matplotlib: Not available')

try:
    import folium
    print('✅ folium: Available')
except ImportError:
    print('❌ folium: Not available')
"

echo ""
echo "🎉 Setup completed!"
echo ""
echo "� Virtual environment info:"
echo "   Vị trí: $(pwd)/$VENV_DIR"
echo "   Python: $VENV_DIR/bin/python"
echo "   Pip: $VENV_DIR/bin/pip"
echo ""
echo "🚀 Cách sử dụng:"
echo "   # Kích hoạt virtual environment"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "   # Chạy applications"
echo "   cd system/"
echo "   python enhanced_map_gui.py    # Enhanced desktop GUI"
echo "   python web_map_interface.py   # Web-based interface"
echo ""
echo "   # Thoát virtual environment"
echo "   deactivate"
echo ""
echo "💡 Nếu GUI vẫn không hoạt động:"
echo "   - Sử dụng web interface (luôn hoạt động)"
echo "   - Thử SSH với X11 forwarding: ssh -X"
echo "   - Sử dụng VNC hoặc remote desktop"
echo ""
echo "🔄 Để sử dụng lại sau này:"
echo "   source $VENV_DIR/bin/activate && cd system/"

# Tạo activation script
echo "💾 Tạo activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
echo "🌍 Kích hoạt Smart Waste Management Environment..."
source venv_waste_management/bin/activate
echo "✅ Virtual environment đã được kích hoạt!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo ""
echo "🚀 Available commands:"
echo "   cd system/                    # Chuyển đến thư mục system"
echo "   python enhanced_map_gui.py    # Desktop GUI"  
echo "   python web_map_interface.py   # Web interface"
echo "   deactivate                    # Thoát virtual environment"
EOF

chmod +x activate_env.sh
echo "✅ Tạo activate_env.sh script - chạy './activate_env.sh' để kích hoạt nhanh!"
