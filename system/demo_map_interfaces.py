#!/usr/bin/env python3
"""
Demo Enhanced Map Interfaces
Test tất cả các giao diện bản đồ đã tạo

Author: Smart Waste Management System
Date: August 2025
"""

import os
import sys
import time
import subprocess
from typing import List, Tuple

# Add system directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_routing_system import SmartRoutingSystem, create_sample_data


def test_matplotlib_backend():
    """Test matplotlib backend availability"""
    print("🧪 Testing matplotlib backends...")
    
    try:
        import matplotlib
        backends = ['TkAgg', 'Qt5Agg', 'Agg']
        available = []
        
        for backend in backends:
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt
                fig = plt.figure()
                plt.close(fig)
                available.append(backend)
            except Exception as e:
                print(f"   ❌ {backend}: {e}")
        
        print(f"✅ Available backends: {available}")
        return available
        
    except ImportError:
        print("❌ matplotlib not available")
        return []


def test_folium_availability():
    """Test folium availability"""
    try:
        import folium
        print("✅ folium: Available")
        return True
    except ImportError:
        print("❌ folium: Not available")
        return False


def demo_basic_interactive():
    """Demo basic interactive map"""
    print("\n🗺️ Demo 1: Basic Interactive Map")
    try:
        from interactive_map import main as interactive_main
        interactive_main()
    except Exception as e:
        print(f"❌ Basic interactive failed: {e}")


def demo_enhanced_gui():
    """Demo enhanced GUI map"""
    print("\n🌟 Demo 2: Enhanced GUI Map")
    try:
        from enhanced_map_gui import main as enhanced_main
        enhanced_main()
    except Exception as e:
        print(f"❌ Enhanced GUI failed: {e}")


def demo_web_interface():
    """Demo web interface"""
    print("\n🌐 Demo 3: Web Interface")
    try:
        from web_map_interface import main as web_main
        map_path, mobile_path = web_main()
        return map_path, mobile_path
    except Exception as e:
        print(f"❌ Web interface failed: {e}")
        return None, None


def show_menu():
    """Hiển thị menu lựa chọn"""
    print("\n" + "="*60)
    print("🗺️ SMART WASTE MANAGEMENT - MAP INTERFACE DEMO")
    print("="*60)
    print()
    print("Chọn giao diện bản đồ:")
    print("1. 📱 Basic Interactive Map (matplotlib)")
    print("2. 🌟 Enhanced GUI Map (như Google Maps)")
    print("3. 🌐 Web-based Map (browser)")
    print("4. 📋 Test tất cả")
    print("5. 🔧 Kiểm tra system requirements")
    print("6. 🚀 Setup GUI cho Linux")
    print("0. ❌ Thoát")
    print()


def check_requirements():
    """Kiểm tra system requirements"""
    print("\n🔍 SYSTEM REQUIREMENTS CHECK")
    print("="*50)
    
    # Python version
    print(f"🐍 Python: {sys.version}")
    
    # Display environment
    display = os.environ.get('DISPLAY', 'Not set')
    print(f"🖥️ DISPLAY: {display}")
    
    # Test imports
    packages = [
        ('matplotlib', 'Required for desktop GUI'),
        ('numpy', 'Required for calculations'),
        ('folium', 'Required for web maps'),
        ('webbrowser', 'Built-in module'),
        ('tkinter', 'GUI backend')
    ]
    
    print("\n📦 Package Status:")
    for package, description in packages:
        try:
            __import__(package)
            print(f"   ✅ {package}: Available - {description}")
        except ImportError:
            print(f"   ❌ {package}: Missing - {description}")
    
    # Test matplotlib backends
    available_backends = test_matplotlib_backend()
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not available_backends:
        print("   - Install matplotlib: pip install matplotlib")
    
    if 'TkAgg' not in available_backends and 'Qt5Agg' not in available_backends:
        print("   - Install GUI backend:")
        print("     Ubuntu/Debian: sudo apt-get install python3-tk")
        print("     Or: pip install PyQt5")
    
    if not test_folium_availability():
        print("   - Install folium for web maps: pip install folium")
    
    if display == 'Not set':
        print("   - For SSH: use 'ssh -X' for X11 forwarding")
        print("   - For WSL: install VcXsrv or X410")
        print("   - Alternative: use web interface")


def setup_linux_gui():
    """Chạy setup script cho Linux"""
    print("\n🐧 Setting up Linux GUI...")
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'setup_linux_gui.sh')
    
    if os.path.exists(script_path):
        try:
            result = subprocess.run(['bash', script_path], check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Setup failed: {e}")
            print(f"Error output: {e.stderr}")
    else:
        print(f"❌ Setup script not found: {script_path}")


def main():
    """Main function"""
    while True:
        show_menu()
        
        try:
            choice = input("Nhập lựa chọn (0-6): ").strip()
            
            if choice == '0':
                print("👋 Tạm biệt!")
                break
                
            elif choice == '1':
                demo_basic_interactive()
                
            elif choice == '2':
                demo_enhanced_gui()
                
            elif choice == '3':
                map_path, mobile_path = demo_web_interface()
                if map_path:
                    print(f"📁 Web map: {map_path}")
                if mobile_path:
                    print(f"📱 Mobile app: {mobile_path}")
                
            elif choice == '4':
                print("🧪 Testing all interfaces...")
                demo_basic_interactive()
                time.sleep(2)
                demo_enhanced_gui()
                time.sleep(2)
                demo_web_interface()
                
            elif choice == '5':
                check_requirements()
                
            elif choice == '6':
                setup_linux_gui()
                
            else:
                print("❌ Lựa chọn không hợp lệ!")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")
        
        input("\n👆 Nhấn Enter để tiếp tục...")


if __name__ == "__main__":
    main()
