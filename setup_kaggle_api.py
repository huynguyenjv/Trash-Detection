"""
Script thiết lập Kaggle API cho Windows
Hướng dẫn từng bước thiết lập Kaggle API credentials

Author: Huy Nguyen
Date: August 2025
"""

import os
import json
from pathlib import Path


def main():
    print("🔑 THIẾT LẬP KAGGLE API")
    print("=" * 40)
    print()
    
    # Kiểm tra xem đã có file chưa
    kaggle_locations = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json"
    ]
    
    existing_file = None
    for location in kaggle_locations:
        if location.exists():
            existing_file = location
            break
    
    if existing_file:
        print(f"✅ Kaggle API đã được thiết lập tại: {existing_file}")
        print()
        
        # Kiểm tra nội dung file
        try:
            with open(existing_file, 'r') as f:
                config = json.load(f)
            if 'username' in config and 'key' in config:
                print(f"👤 Username: {config['username']}")
                print("🔐 API Key: ********")
                print()
                print("✅ Cấu hình hợp lệ!")
                return
        except:
            print("❌ File không hợp lệ, cần thiết lập lại")
    
    print("❌ Chưa có Kaggle API key")
    print()
    print("📋 HƯỚNG DẪN THIẾT LẬP:")
    print("1. Mở trình duyệt và đi đến: https://kaggle.com")
    print("2. Đăng nhập vào tài khoản Kaggle của bạn")
    print("3. Vào Account Settings (click vào avatar → Account)")
    print("4. Cuộn xuống phần 'API'")
    print("5. Nhấn 'Create New API Token'")
    print("6. File 'kaggle.json' sẽ được download")
    print("7. Quay lại script này để hoàn thành thiết lập")
    print()
    
    choice = input("Bạn đã có file kaggle.json chưa? (y/n): ").lower()
    
    if choice == 'y':
        # Hỏi đường dẫn file
        print("\n📁 Nhập đường dẫn đến file kaggle.json:")
        print("(Hoặc nhấn Enter để nhập thông tin thủ công)")
        file_path = input("Đường dẫn: ").strip()
        
        if file_path and Path(file_path).exists():
            setup_from_file(file_path)
        else:
            setup_manual()
    else:
        print("\n💡 Sau khi tải file kaggle.json từ Kaggle, hãy chạy lại script này")


def setup_from_file(file_path):
    """Thiết lập từ file kaggle.json có sẵn"""
    try:
        # Đọc file
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        if 'username' not in config or 'key' not in config:
            print("❌ File không hợp lệ - thiếu username hoặc key")
            return
        
        # Chọn vị trí lưu (ưu tiên ~/.kaggle/)
        target_dir = Path.home() / ".kaggle"
        target_file = target_dir / "kaggle.json"
        
        # Tạo thư mục
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        with open(target_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set permissions trên Unix
        if os.name != 'nt':
            os.chmod(target_file, 0o600)
        
        print(f"\n✅ Đã thiết lập Kaggle API tại: {target_file}")
        print(f"👤 Username: {config['username']}")
        print("🔐 API Key: ********")
        print("\n🎉 Thiết lập hoàn tất! Bạn có thể sử dụng Kaggle API.")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")


def setup_manual():
    """Thiết lập thủ công"""
    print("\n✋ THIẾT LẬP THỦ CÔNG")
    print("Nhập thông tin Kaggle API của bạn:")
    print()
    
    username = input("Username: ").strip()
    if not username:
        print("❌ Username không được để trống")
        return
    
    api_key = input("API Key: ").strip()
    if not api_key:
        print("❌ API Key không được để trống")
        return
    
    # Tạo config
    config = {
        "username": username,
        "key": api_key
    }
    
    # Chọn vị trí lưu
    target_dir = Path.home() / ".kaggle"
    target_file = target_dir / "kaggle.json"
    
    # Tạo thư mục
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu file
    with open(target_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set permissions
    if os.name != 'nt':
        os.chmod(target_file, 0o600)
    
    print(f"\n✅ Đã thiết lập Kaggle API tại: {target_file}")
    print("\n🎉 Thiết lập hoàn tất! Bạn có thể sử dụng Kaggle API.")


def test_kaggle_api():
    """Test Kaggle API"""
    print("\n🧪 KIỂM TRA KAGGLE API")
    print("-" * 30)
    
    try:
        import kaggle
        
        # Test authentication
        kaggle.api.authenticate()
        print("✅ Xác thực thành công")
        
        # Test một API call đơn giản
        user = kaggle.api.get_user()
        print(f"✅ Kết nối thành công với user: {user}")
        
    except ImportError:
        print("❌ Package 'kaggle' chưa được cài đặt")
        print("Cài đặt: pip install kaggle")
    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    main()
    
    # Hỏi có muốn test không
    print()
    test_choice = input("Bạn có muốn test Kaggle API không? (y/n): ").lower()
    if test_choice == 'y':
        test_kaggle_api()
    
    print("\n📚 Xem thêm hướng dẫn tại:")
    print("https://github.com/Kaggle/kaggle-api#api-credentials")
