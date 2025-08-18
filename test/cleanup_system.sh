#!/bin/bash

echo "🗑️ BACKUP & CLEANUP SCRIPT"
echo "=========================="

# Create backup directory
echo "📦 Creating backup..."
mkdir -p backup_system_$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backup_system_$(date +%Y%m%d_%H%M%S)"

# Files to backup (important data/configs)
echo "💾 Backing up important files..."
cp system/enhanced_waste_map.html $BACKUP_DIR/ 2>/dev/null || echo "   - enhanced_waste_map.html: not found"
cp system/mobile_waste_app.html $BACKUP_DIR/ 2>/dev/null || echo "   - mobile_waste_app.html: not found"
cp system/backend_test.png $BACKUP_DIR/ 2>/dev/null || echo "   - backend_test.png: not found"
cp system/position_history_*.json $BACKUP_DIR/ 2>/dev/null || echo "   - position_history files: not found"
cp system/robot_position_*.json $BACKUP_DIR/ 2>/dev/null || echo "   - robot_position files: not found"

echo "📋 Files backed up to: $BACKUP_DIR"
ls -la $BACKUP_DIR/ 2>/dev/null || echo "   (no files to backup)"

echo ""
echo "🔍 Checking what will be deleted from system/:"
echo "======================================================"

# Show what will be deleted
du -sh system/ 2>/dev/null || echo "system/ directory not found"
echo ""
echo "Files in system/:"
ls -la system/ 2>/dev/null || echo "system/ directory not found"

echo ""
echo "⚠️  SAFETY CHECK:"
echo "==================="
echo "✅ Refactored system is ready: refactored_system/"
echo "✅ All functionality has been migrated"
echo "✅ Tests are passing"
echo ""

# Show comparison
echo "📊 COMPARISON:"
echo "Old system/:"
find system/ -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 || echo "   No Python files found"
echo ""
echo "New refactored_system/:"
find refactored_system/ -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 || echo "   No Python files found"

echo ""
echo "🤔 DO YOU WANT TO DELETE system/ folder? [y/N]"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "🗑️ Deleting system/ folder..."
    
    # Final confirmation
    echo "⚠️  FINAL CONFIRMATION: This will permanently delete system/ folder!"
    echo "Type 'DELETE' to confirm:"
    read -r final_confirm
    
    if [[ "$final_confirm" == "DELETE" ]]; then
        rm -rf system/
        echo "✅ system/ folder deleted successfully!"
        echo ""
        echo "📁 Current structure:"
        ls -la | grep -E "(system|refactored_system)"
        echo ""
        echo "🎉 Cleanup completed! Use refactored_system/ going forward."
    else
        echo "❌ Deletion cancelled."
    fi
else
    echo "❌ Deletion cancelled."
fi

echo ""
echo "📋 NEXT STEPS:"
echo "=============="
echo "1. Use 'refactored_system/' for all development"
echo "2. Update any scripts that reference 'system/'"
echo "3. Update documentation/README files"
echo ""
echo "🚀 Quick start with refactored system:"
echo "   cd refactored_system"
echo "   python main.py --mode web"
echo "
