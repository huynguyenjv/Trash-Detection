#!/usr/bin/env python3
"""
Thiết lập và kiểm tra GUI cho Linux
Tự động cài đặt backend phù hợp cho matplotlib trên Linux

Author: Smart Waste Management System
Date: August 2025
"""

import os
import sys
import subprocess
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Dict


class LinuxGUIChecker:
    """Kiểm tra và thiết lập GUI cho Linux"""
    
    def __init__(self):
        self.available_backends = []
        self.recommended_backend = None
        self.display_available = 'DISPLAY' in os.environ
    
    def check_display_server(self) -> Dict[str, bool]:
        """Kiểm tra display server"""
        result = {
            'X11': False,
            'Wayland': False,
            'DISPLAY_env': self.display_available
        }
        
        # Kiểm tra X11
        if self.display_available:
            try:
                subprocess.run(['xset', 'q'], 
                             capture_output=True, check=True, timeout=5)
                result['X11'] = True
            except:
                pass
        
        # Kiểm tra Wayland
        if 'WAYLAND_DISPLAY' in os.environ:
            result['Wayland'] = True
            
        return result
    
    def check_gui_packages(self) -> Dict[str, bool]:
        """Kiểm tra các package GUI có sẵn"""
        packages = {
            'tkinter': False,
            'PyQt5': False,
            'PyQt6': False,
            'PySide2': False,
            'PySide6': False,
            'gtk': False
        }
        
        # Test tkinter
        try:
            import tkinter
            packages['tkinter'] = True
        except ImportError:
            pass
            
        # Test PyQt5
        try:
            import PyQt5
            packages['PyQt5'] = True
        except ImportError:
            pass
            
        # Test PyQt6
        try:
            import PyQt6
            packages['PyQt6'] = True
        except ImportError:
            pass
            
        # Test PySide2
        try:
            import PySide2
            packages['PySide2'] = True
        except ImportError:
            pass
            
        # Test PySide6
        try:
            import PySide6
            packages['PySide6'] = True
        except ImportError:
            pass
        
        return packages
    
    def get_available_backends(self) -> List[str]:
        """Lấy danh sách backend có thể dùng"""
        all_backends = [
            'TkAgg',      # Tkinter
            'Qt5Agg',     # PyQt5/PySide2  
            'Qt6Agg',     # PyQt6/PySide6
            'GTK3Agg',    # GTK3
            'GTK4Agg',    # GTK4
            'Agg'         # Non-interactive (fallback)
        ]
        
        available = []
        
        for backend in all_backends:
            try:
                matplotlib.use(backend, force=True)
                fig = plt.figure()
                plt.close(fig)
                available.append(backend)
            except Exception:
                pass
                
        return available
    
    def recommend_backend(self) -> str:
        """Đề xuất backend tốt nhất"""
        display_info = self.check_display_server()
        gui_packages = self.check_gui_packages()
        
        if not display_info['DISPLAY_env']:
            return 'Agg'  # No display server
            
        # Priority order for interactive backends
        if gui_packages['tkinter']:
            return 'TkAgg'
        elif gui_packages['PyQt5']:
            return 'Qt5Agg'
        elif gui_packages['PyQt6']:
            return 'Qt6Agg'
        elif gui_packages['PySide2']:
            return 'Qt5Agg'
        elif gui_packages['PySide6']:
            return 'Qt6Agg'
        else:
            return 'Agg'  # Fallback
    
    def install_gui_package(self, package: str) -> bool:
        """Cài đặt GUI package"""
        install_commands = {
            'tkinter': [
                'sudo apt-get update',
                'sudo apt-get install python3-tk -y'
            ],
            'PyQt5': [
                'pip install PyQt5'
            ],
            'PyQt6': [
                'pip install PyQt6'
            ]
        }
        
        if package not in install_commands:
            return False
            
        try:
            for cmd in install_commands[package]:
                result = subprocess.run(
                    cmd.split(), 
                    capture_output=True, 
                    text=True,
                    timeout=300
                )
                if result.returncode != 0:
                    print(f"❌ Failed: {cmd}")
                    print(f"Error: {result.stderr}")
                    return False
                else:
                    print(f"✅ Success: {cmd}")
            return True
        except subprocess.TimeoutExpired:
            print("⏱️  Installation timeout")
            return False
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
    
    def generate_report(self) -> str:
        """Tạo báo cáo tình trạng GUI"""
        display_info = self.check_display_server()
        gui_packages = self.check_gui_packages()
        available_backends = self.get_available_backends()
        recommended = self.recommend_backend()
        
        report = [
            "🖥️  LINUX GUI STATUS REPORT",
            "=" * 50,
            "",
            "📺 Display Server:",
            f"  - DISPLAY environment: {'✅' if display_info['DISPLAY_env'] else '❌'}",
            f"  - X11: {'✅' if display_info['X11'] else '❌'}",
            f"  - Wayland: {'✅' if display_info['Wayland'] else '❌'}",
            "",
            "📦 GUI Packages:",
        ]
        
        for package, available in gui_packages.items():
            status = '✅' if available else '❌'
            report.append(f"  - {package}: {status}")
        
        report.extend([
            "",
            "🎨 Available Matplotlib Backends:",
        ])
        
        for backend in available_backends:
            marker = '🎯' if backend == recommended else '  '
            report.append(f"  {marker} {backend}")
            
        report.extend([
            "",
            f"🔧 Recommended Backend: {recommended}",
            "",
            "💡 Solutions:",
        ])
        
        if not display_info['DISPLAY_env']:
            report.append("  - No display server detected")
            report.append("  - Use 'Agg' backend for image generation")
            report.append("  - Or run with X11 forwarding: ssh -X")
        elif not gui_packages['tkinter'] and not any(gui_packages.values()):
            report.append("  - Install GUI toolkit:")
            report.append("    sudo apt-get install python3-tk")
            report.append("    # OR")
            report.append("    pip install PyQt5")
        elif recommended == 'Agg':
            report.append("  - Install interactive backend:")
            report.append("    sudo apt-get install python3-tk")
        else:
            report.append("  - GUI setup looks good! ✅")
            
        return "\n".join(report)


def main():
    """Main function"""
    print("🐧 Linux GUI Checker for Trash Detection System")
    print("=" * 60)
    
    checker = LinuxGUIChecker()
    
    # Generate and display report
    report = checker.generate_report()
    print(report)
    
    # Interactive setup
    recommended = checker.recommend_backend()
    
    if recommended == 'Agg':
        print("\n⚠️  WARNING: Only non-interactive backend available!")
        print("Interactive map will be saved as image file.")
        
        # Offer to install GUI package
        if checker.check_display_server()['DISPLAY_env']:
            response = input("\n🤔 Install tkinter for interactive GUI? [y/N]: ")
            if response.lower() in ['y', 'yes']:
                print("\n📦 Installing tkinter...")
                success = checker.install_gui_package('tkinter')
                if success:
                    print("✅ Installation completed! Please restart your application.")
                else:
                    print("❌ Installation failed. Using image output mode.")
    
    # Test the recommended backend
    print(f"\n🧪 Testing backend: {recommended}")
    try:
        matplotlib.use(recommended, force=True)
        fig = plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title(f"Test plot with {recommended} backend")
        
        if recommended == 'Agg':
            plt.savefig('backend_test.png')
            print("✅ Backend test successful! Image saved as 'backend_test.png'")
        else:
            print("✅ Backend test successful! Close the plot window to continue.")
            plt.show()
            
        plt.close(fig)
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
    
    # Create config for the system
    config_content = f"""# Auto-generated matplotlib backend config
# Generated by Linux GUI Checker

RECOMMENDED_BACKEND = '{recommended}'
HAS_DISPLAY = {checker.display_available}
INTERACTIVE_MODE = {recommended != 'Agg'}

def setup_matplotlib():
    import matplotlib
    matplotlib.use('{recommended}')
    return '{recommended}'
"""
    
    with open('gui_config.py', 'w') as f:
        f.write(config_content)
    
    print("💾 Configuration saved to 'gui_config.py'")
    print("✅ Setup completed!")


if __name__ == "__main__":
    main()
