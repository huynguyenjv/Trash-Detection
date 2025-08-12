#!/usr/bin/env python3
"""
Guide to setup Kaggle API and download dataset
"""

import os

def setup_kaggle_guide():
    print("🔑 HƯỚNG DẪN SETUP KAGGLE API")
    print("=" * 50)
    print("1. Đăng nhập vào Kaggle.com")
    print("2. Vào Account Settings (Click avatar > Account)")
    print("3. Scroll xuống phần 'API', click 'Create New Token'")
    print("4. Tải file kaggle.json về máy")
    print("5. Chạy lệnh sau để setup:")
    print()
    print("   mkdir -p ~/.config/kaggle/")
    print("   mv ~/Downloads/kaggle.json ~/.config/kaggle/")
    print("   chmod 600 ~/.config/kaggle/kaggle.json")
    print()
    print("6. Test bằng lệnh: kaggle datasets list")
    print()
    
    # Check if already setup
    kaggle_path = os.path.expanduser("~/.config/kaggle/kaggle.json")
    if os.path.exists(kaggle_path):
        print("✅ Kaggle API đã được setup!")
        print("💾 Dataset sẽ được tải về tự động")
    else:
        print("⚠️  Kaggle API chưa setup")
        print("📁 Hoặc bạn có thể tự tải dataset từ:")
        print("   https://www.kaggle.com/datasets/mostafaabla/garbage-classification-v2")
        print("   và giải nén vào thư mục data/raw/")

def create_sample_structure():
    """Create sample directory structure if no real data"""
    print("\n📁 TẠO CẤU TRÚC THƯ MỤC MẪU")
    print("=" * 30)
    
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    for split in ['train', 'test']:
        for category in categories:
            dir_path = f"data/raw/garbage_classification_v2/{split}/{category}"
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Created: {dir_path}")
    
    print("\n📝 Bạn có thể:")
    print("1. Tải dataset thật từ Kaggle")
    print("2. Hoặc thêm ảnh của bạn vào các thư mục trên")
    print("3. Chạy lại pipeline để train model")

if __name__ == "__main__":
    setup_kaggle_guide()
    create_sample_structure()
