"""
Utility script để lấy và hiển thị tọa độ hiện tại trong hệ thống
Có thể chạy độc lập để kiểm tra thông tin định vị

Author: Smart Waste Management System  
Date: August 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_routing_system import SmartRoutingSystem, GPSCoordinate, create_sample_data
import json
import time


def get_current_position_info(routing_system: SmartRoutingSystem) -> dict:
    """
    Lấy thông tin chi tiết về vị trí hiện tại
    
    Args:
        routing_system: Hệ thống định tuyến
        
    Returns:
        Dict chứa thông tin vị trí
    """
    if not routing_system.current_position:
        return {
            "status": "NO_POSITION",
            "message": "Chưa set vị trí robot",
            "position": None
        }
    
    pos = routing_system.current_position
    
    # Tính khoảng cách đến các bãi rác
    from smart_routing_system import HaversineCalculator
    
    bin_distances = []
    for bin_id, bin_obj in routing_system.waste_bins.items():
        distance = HaversineCalculator.distance(pos, bin_obj.location)
        bin_distances.append({
            "bin_id": bin_id,
            "distance_km": round(distance, 3),
            "status": bin_obj.status.value,
            "capacity_ratio": round(bin_obj.capacity_ratio, 2),
            "supported_types": [wt.value for wt in bin_obj.supported_types]
        })
    
    # Sắp xếp theo khoảng cách
    bin_distances.sort(key=lambda x: x["distance_km"])
    
    return {
        "status": "ACTIVE",
        "position": {
            "latitude": pos.lat,
            "longitude": pos.lng,
            "coordinates_string": f"{pos.lat:.6f}, {pos.lng:.6f}"
        },
        "nearby_bins": bin_distances[:3],  # 3 bãi gần nhất
        "all_bins_count": len(bin_distances),
        "timestamp": time.time()
    }


def display_position_info(info: dict, detailed: bool = False):
    """Hiển thị thông tin vị trí"""
    print("🗺️ THÔNG TIN VỊ TRÍ HIỆN TẠI")
    print("=" * 50)
    
    if info["status"] == "NO_POSITION":
        print("❌ " + info["message"])
        return
    
    pos = info["position"]
    print(f"📍 Tọa độ hiện tại:")
    print(f"   Latitude (Vĩ độ):  {pos['latitude']:.6f}")
    print(f"   Longitude (Kinh độ): {pos['longitude']:.6f}")
    print(f"   Chuỗi tọa độ: {pos['coordinates_string']}")
    
    print(f"\n🏢 Bãi rác gần nhất:")
    for i, bin_info in enumerate(info["nearby_bins"], 1):
        status_icon = "🟢" if bin_info["status"] == "OK" else "🟡" if bin_info["status"] == "NEAR_FULL" else "🔴"
        print(f"   {i}. {bin_info['bin_id']} - {bin_info['distance_km']}km {status_icon}")
        if detailed:
            print(f"      Trạng thái: {bin_info['status']}")
            print(f"      Độ đầy: {bin_info['capacity_ratio']*100:.0f}%")
            print(f"      Hỗ trợ: {', '.join(bin_info['supported_types'])}")
    
    print(f"\n📊 Tổng số bãi rác: {info['all_bins_count']}")
    print(f"⏰ Thời gian: {time.strftime('%H:%M:%S %d/%m/%Y', time.localtime(info['timestamp']))}")


def update_robot_position(routing_system: SmartRoutingSystem, lat: float, lng: float):
    """Cập nhật vị trí robot"""
    new_pos = GPSCoordinate(lat, lng)
    routing_system.update_robot_position(new_pos)
    
    print(f"✅ Đã cập nhật vị trí robot:")
    print(f"   Latitude: {lat}")
    print(f"   Longitude: {lng}")


def save_position_to_file(info: dict, filename: str = None):
    """Lưu thông tin vị trí vào file"""
    if not filename:
        timestamp = int(time.time())
        filename = f"current_position_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Đã lưu thông tin vị trí: {filename}")


def load_position_from_file(filename: str) -> dict:
    """Load thông tin vị trí từ file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def demo_position_commands():
    """Demo các lệnh quản lý vị trí"""
    print("🚀 DEMO QUẢN LÝ VỊ TRÍ ROBOT")
    print("=" * 50)
    
    # Tạo hệ thống mẫu
    system = create_sample_data()
    
    # Hiển thị vị trí ban đầu
    print("\n1️⃣ Vị trí ban đầu:")
    info = get_current_position_info(system)
    display_position_info(info)
    
    # Cập nhật vị trí mới
    print("\n2️⃣ Cập nhật vị trí mới:")
    update_robot_position(system, 10.780000, 106.690000)  # Di chuyển đến vị trí khác
    
    # Hiển thị vị trí mới
    print("\n3️⃣ Vị trí sau khi cập nhật:")
    info_new = get_current_position_info(system)
    display_position_info(info_new, detailed=True)
    
    # Lưu vị trí
    print("\n4️⃣ Lưu thông tin vị trí:")
    save_position_to_file(info_new)
    
    # Test các vị trí khác
    test_positions = [
        (10.762622, 106.660172, "Gần BIN001"),
        (10.775831, 106.700806, "Gần BIN002"), 
        (10.745567, 106.690123, "Gần BIN004")
    ]
    
    print("\n5️⃣ Test các vị trí khác:")
    for lat, lng, desc in test_positions:
        print(f"\n📍 {desc}:")
        update_robot_position(system, lat, lng)
        info = get_current_position_info(system)
        print(f"   Bãi gần nhất: {info['nearby_bins'][0]['bin_id']} ({info['nearby_bins'][0]['distance_km']}km)")


def interactive_position_setter():
    """Chế độ tương tác để set vị trí"""
    print("🎯 CHẾ độ TƯƠNG TÁC - THIẾT LẬP VỊ TRÍ ROBOT")
    print("=" * 50)
    
    system = create_sample_data()
    
    while True:
        print("\n📋 Lựa chọn:")
        print("1. Hiển thị vị trí hiện tại")
        print("2. Cập nhật vị trí mới")
        print("3. Lưu vị trí vào file")
        print("4. Load vị trí từ file")
        print("5. Hiển thị tất cả bãi rác")
        print("0. Thoát")
        
        choice = input("\n👉 Nhập lựa chọn (0-5): ").strip()
        
        if choice == "0":
            print("👋 Tạm biệt!")
            break
        
        elif choice == "1":
            info = get_current_position_info(system)
            display_position_info(info, detailed=True)
        
        elif choice == "2":
            try:
                lat = float(input("📍 Nhập Latitude (vĩ độ): "))
                lng = float(input("📍 Nhập Longitude (kinh độ): "))
                update_robot_position(system, lat, lng)
                
                # Hiển thị vị trí mới
                info = get_current_position_info(system)
                display_position_info(info)
                
            except ValueError:
                print("❌ Tọa độ không hợp lệ!")
        
        elif choice == "3":
            info = get_current_position_info(system)
            if info["status"] == "ACTIVE":
                save_position_to_file(info)
            else:
                print("❌ Chưa có vị trí để lưu!")
        
        elif choice == "4":
            filename = input("📁 Nhập tên file: ").strip()
            try:
                info = load_position_from_file(filename)
                print(f"✅ Đã load thông tin từ {filename}")
                display_position_info(info, detailed=True)
            except FileNotFoundError:
                print("❌ Không tìm thấy file!")
            except json.JSONDecodeError:
                print("❌ File không đúng định dạng!")
        
        elif choice == "5":
            print("\n🏢 TẤT CẢ BÃI RÁC TRONG HỆ THỐNG:")
            for bin_id, bin_obj in system.waste_bins.items():
                status_icon = "🟢" if bin_obj.status.value == "OK" else "🟡" if bin_obj.status.value == "NEAR_FULL" else "🔴"
                print(f"   {bin_id}: {bin_obj.location.lat:.6f}, {bin_obj.location.lng:.6f} {status_icon}")
                print(f"      Trạng thái: {bin_obj.status.value}")
                print(f"      Sức chứa: {bin_obj.current_capacity}/{bin_obj.max_capacity}kg")
                print(f"      Hỗ trợ: {', '.join([wt.value for wt in bin_obj.supported_types])}")
                print()
        
        else:
            print("❌ Lựa chọn không hợp lệ!")


def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Position Management Utility')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--show', action='store_true', help='Show current position')
    parser.add_argument('--lat', type=float, help='Set latitude')
    parser.add_argument('--lng', type=float, help='Set longitude')
    parser.add_argument('--save', type=str, help='Save position to file')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_position_commands()
    elif args.interactive:
        interactive_position_setter()
    elif args.show or args.lat or args.lng or args.save:
        system = create_sample_data()
        
        if args.lat and args.lng:
            update_robot_position(system, args.lat, args.lng)
        
        info = get_current_position_info(system)
        display_position_info(info, detailed=True)
        
        if args.save:
            save_position_to_file(info, args.save)
    
    else:
        print("🗺️ Position Management Utility")
        print("Usage examples:")
        print("  python position_utils.py --demo")
        print("  python position_utils.py --interactive") 
        print("  python position_utils.py --show")
        print("  python position_utils.py --lat 10.77 --lng 106.68")
        print("  python position_utils.py --show --save position.json")


if __name__ == "__main__":
    main()
