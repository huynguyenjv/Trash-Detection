#!/bin/bash
"""
Setup Script cho Smart Waste Management System
Tạo một environment thống nhất cho cả training và system

Author: Smart Waste Management Team
Date: August 2025
"""

echo "🚀 SETUP SMART WASTE MANAGEMENT SYSTEM"
echo "======================================"

# Check if running in existing environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Đang sử dụng virtual environment: $VIRTUAL_ENV"
    USE_EXISTING_ENV=true
else
    echo "⚠️ Không có virtual environment active"
    USE_EXISTING_ENV=false
fi

# Function to create new environment
create_new_env() {
    echo "📦 Tạo environment mới: smart_waste_env"
    python3 -m venv smart_waste_env
    source smart_waste_env/bin/activate
    
    echo "⬆️ Upgrade pip"
    python -m pip install --upgrade pip
}

# Function to install packages
install_packages() {
    echo "📥 Cài đặt packages cần thiết..."
    
    # Core packages
    echo "1. Installing core packages..."
    pip install numpy pandas matplotlib opencv-python Pillow pyyaml tqdm psutil python-dateutil requests
    
    # ML packages  
    echo "2. Installing ML packages..."
    pip install torch torchvision ultralytics scikit-learn seaborn
    
    # Web interface
    echo "3. Installing web packages..."
    pip install folium
    
    # Development tools
    echo "4. Installing dev tools..."
    pip install pytest black flake8
    
    echo "✅ Tất cả packages đã được cài đặt!"
}

# Main setup logic
if [[ "$USE_EXISTING_ENV" == true ]]; then
    echo "🔄 Sử dụng environment hiện tại và cài đặt thêm packages..."
    install_packages
else
    echo "🆕 Tạo environment mới..."
    create_new_env
    install_packages
fi

echo ""
echo "✅ SETUP HOÀN THÀNH!"
echo ""
echo "📋 Để sử dụng hệ thống:"
echo ""
echo "1. Training YOLO model:"
echo "   cd src/"
echo "   python train.py"
echo "   python evaluate.py --model ../models/best.pt"
echo ""
echo "2. Smart Waste System:"
echo "   cd system/"
echo "   python main.py --mode web"
echo "   python enhanced_main.py --mode demo"
echo ""  
echo "3. Test complete system:"
echo "   python system/test_smart_system.py"
echo ""
echo "🎉 Happy coding!"
