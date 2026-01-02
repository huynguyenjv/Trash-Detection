# -*- coding: utf-8 -*-
"""
Seed script to populate WasteBin table with HCM waste locations data
Run this script after database is created to add initial data
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session
from app.database import engine, SessionLocal
from app.models import Base, WasteBin
from app.data.hcm_waste_locations import HCM_WASTE_LOCATIONS
import random


def map_waste_type_to_category(waste_types: list) -> str:
    """Map location waste types to WasteBin category"""
    if "hazardous" in waste_types:
        return "hazardous"
    elif "recyclable" in waste_types and "organic" not in waste_types:
        return "recyclable"
    elif "organic" in waste_types and "recyclable" not in waste_types:
        return "organic"
    else:
        return "other"


def seed_waste_bins():
    """Seed WasteBin table with HCM waste locations"""
    
    print("🌱 Starting to seed waste bins data...")
    
    # Create tables if not exist
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Check existing bins
        existing_count = db.query(WasteBin).count()
        print(f"📊 Existing waste bins in database: {existing_count}")
        
        added_count = 0
        skipped_count = 0
        
        for location in HCM_WASTE_LOCATIONS:
            # Check if already exists by name
            existing = db.query(WasteBin).filter(WasteBin.name == location["name"]).first()
            
            if existing:
                print(f"⏭️  Skipping (exists): {location['name']}")
                skipped_count += 1
                continue
            
            # Determine category from waste_types
            category = map_waste_type_to_category(location.get("waste_types", ["other"]))
            
            # Create new waste bin
            waste_bin = WasteBin(
                name=location["name"],
                category=category,
                latitude=location["latitude"],
                longitude=location["longitude"],
                address=location.get("address", ""),
                capacity=location.get("capacity_tons_per_day", 100),
                current_fill=random.uniform(10, 80),  # Random fill level for demo
                is_active=location.get("status", "active") == "active"
            )
            
            db.add(waste_bin)
            added_count += 1
            print(f"✅ Added: {location['name']} ({category})")
        
        db.commit()
        
        print(f"\n{'='*50}")
        print(f"🎉 Seeding completed!")
        print(f"   ➕ Added: {added_count} waste bins")
        print(f"   ⏭️  Skipped: {skipped_count} (already existed)")
        print(f"   📊 Total in database: {db.query(WasteBin).count()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def clear_and_reseed():
    """Clear all waste bins and reseed from scratch"""
    print("⚠️  Clearing all existing waste bins...")
    
    db = SessionLocal()
    try:
        deleted = db.query(WasteBin).delete()
        db.commit()
        print(f"🗑️  Deleted {deleted} waste bins")
    finally:
        db.close()
    
    seed_waste_bins()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        clear_and_reseed()
    else:
        seed_waste_bins()
        print("\n💡 Tip: Run with --clear flag to delete all and reseed")
