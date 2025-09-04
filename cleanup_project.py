#!/usr/bin/env python3
"""
Project Cleanup Script for Trash Detection System
Organizes and cleans up duplicate files, logs, and unnecessary artifacts
"""

import os
import shutil
import glob
from pathlib import Path
import json

class ProjectCleaner:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.cleaned_items = []
        self.errors = []
        
    def log_action(self, action, path, success=True):
        """Log cleanup actions"""
        status = "✅" if success else "❌"
        message = f"{status} {action}: {path}"
        print(message)
        
        if success:
            self.cleaned_items.append({"action": action, "path": str(path)})
        else:
            self.errors.append({"action": action, "path": str(path)})
    
    def clean_python_cache(self):
        """Remove Python cache files and directories"""
        print("\n🧹 Cleaning Python cache files...")
        
        # Find and remove __pycache__ directories
        for pycache_dir in self.project_root.rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir)
                self.log_action("Removed __pycache__", pycache_dir)
            except Exception as e:
                self.log_action(f"Failed to remove __pycache__", pycache_dir, False)
        
        # Remove .pyc files
        for pyc_file in self.project_root.rglob("*.pyc"):
            try:
                pyc_file.unlink()
                self.log_action("Removed .pyc", pyc_file)
            except Exception as e:
                self.log_action("Failed to remove .pyc", pyc_file, False)
    
    def clean_logs(self):
        """Clean up log files but keep recent ones"""
        print("\n📄 Cleaning log files...")
        
        log_extensions = [".log", ".out", ".err"]
        
        for ext in log_extensions:
            for log_file in self.project_root.rglob(f"*{ext}"):
                # Keep logs that are less than 7 days old
                if log_file.stat().st_mtime < (Path().stat().st_mtime - 604800):  # 7 days
                    try:
                        log_file.unlink()
                        self.log_action("Removed old log", log_file)
                    except Exception as e:
                        self.log_action("Failed to remove log", log_file, False)
    
    def organize_requirements(self):
        """Consolidate requirement files"""
        print("\n📦 Organizing requirements files...")
        
        # Find all requirements files
        req_files = list(self.project_root.glob("requirements*.txt"))
        
        if len(req_files) > 1:
            # Create requirements directory
            req_dir = self.project_root / "requirements"
            req_dir.mkdir(exist_ok=True)
            
            for req_file in req_files:
                if req_file.name != "requirements.txt":
                    try:
                        new_path = req_dir / req_file.name
                        shutil.move(str(req_file), str(new_path))
                        self.log_action("Moved to requirements/", req_file)
                    except Exception as e:
                        self.log_action("Failed to move requirements", req_file, False)
    
    def clean_model_files(self):
        """Organize model files"""
        print("\n🤖 Organizing model files...")
        
        # Find scattered model files
        model_extensions = [".pt", ".pth", ".onnx", ".h5"]
        
        for ext in model_extensions:
            for model_file in self.project_root.rglob(f"*{ext}"):
                # Skip if already in models directory
                if "models" not in str(model_file.parent):
                    models_dir = self.project_root / "models"
                    models_dir.mkdir(exist_ok=True)
                    
                    try:
                        new_path = models_dir / model_file.name
                        if not new_path.exists():
                            shutil.move(str(model_file), str(new_path))
                            self.log_action("Moved to models/", model_file)
                    except Exception as e:
                        self.log_action("Failed to move model", model_file, False)
    
    def clean_temp_files(self):
        """Remove temporary files"""
        print("\n🗑️ Cleaning temporary files...")
        
        temp_patterns = [
            "*.tmp", "*.temp", "*~", ".DS_Store", "Thumbs.db",
            "*.bak", "*.swp", "*.swo"
        ]
        
        for pattern in temp_patterns:
            for temp_file in self.project_root.rglob(pattern):
                try:
                    temp_file.unlink()
                    self.log_action("Removed temp file", temp_file)
                except Exception as e:
                    self.log_action("Failed to remove temp", temp_file, False)
    
    def organize_scripts(self):
        """Organize shell scripts"""
        print("\n📜 Organizing scripts...")
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        script_extensions = [".sh", ".bat", ".ps1"]
        
        for ext in script_extensions:
            for script_file in self.project_root.glob(f"*{ext}"):
                try:
                    new_path = scripts_dir / script_file.name
                    if not new_path.exists():
                        shutil.move(str(script_file), str(new_path))
                        self.log_action("Moved to scripts/", script_file)
                except Exception as e:
                    self.log_action("Failed to move script", script_file, False)
    
    def organize_documentation(self):
        """Organize documentation files"""
        print("\n📚 Organizing documentation...")
        
        docs_dir = self.project_root / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Move markdown files (except main README)
        for md_file in self.project_root.glob("*.md"):
            if md_file.name not in ["README.md"]:
                try:
                    new_path = docs_dir / md_file.name
                    if not new_path.exists():
                        shutil.move(str(md_file), str(new_path))
                        self.log_action("Moved to docs/", md_file)
                except Exception as e:
                    self.log_action("Failed to move doc", md_file, False)
    
    def clean_duplicate_files(self):
        """Remove obvious duplicate files"""
        print("\n🔄 Checking for duplicate files...")
        
        # Look for files with numbers or "copy" in name
        duplicates = []
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                name = file_path.stem.lower()
                if any(x in name for x in ["copy", "backup", "old", " (1)", " (2)"]):
                    duplicates.append(file_path)
        
        for dup in duplicates:
            try:
                dup.unlink()
                self.log_action("Removed duplicate", dup)
            except Exception as e:
                self.log_action("Failed to remove duplicate", dup, False)
    
    def create_gitignore(self):
        """Create/update .gitignore file"""
        print("\n📝 Updating .gitignore...")
        
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/
trash_detection_env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/
*.out
*.err

# Model files (large)
*.pt
*.pth
*.h5
*.onnx
models/*.pt
models/*.pth

# Data
data/raw/
data/processed/
*.csv
*.json
datasets/

# Temporary files
*.tmp
*.temp
*.bak

# Node modules (if any)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Testing
.coverage
.pytest_cache/
htmlcov/

# Build artifacts
dist/
build/
"""
        
        gitignore_path = self.project_root / ".gitignore"
        try:
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write(gitignore_content.strip())
            self.log_action("Updated .gitignore", gitignore_path)
        except Exception as e:
            self.log_action("Failed to update .gitignore", gitignore_path, False)
    
    def generate_report(self):
        """Generate cleanup report"""
        print("\n📊 Generating cleanup report...")
        
        report = {
            "timestamp": str(Path().stat().st_mtime),
            "cleaned_items": len(self.cleaned_items),
            "errors": len(self.errors),
            "actions": self.cleaned_items,
            "failed_actions": self.errors
        }
        
        report_path = self.project_root / "cleanup_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            self.log_action("Generated cleanup report", report_path)
        except Exception as e:
            self.log_action("Failed to generate report", report_path, False)
    
    def run_full_cleanup(self):
        """Run complete project cleanup"""
        print("🚀 Starting project cleanup...")
        
        self.clean_python_cache()
        self.clean_temp_files()
        self.clean_logs()
        self.clean_duplicate_files()
        self.organize_requirements()
        self.organize_scripts()
        self.organize_documentation()
        self.clean_model_files()
        self.create_gitignore()
        self.generate_report()
        
        print(f"\n✨ Cleanup completed!")
        print(f"📈 Actions taken: {len(self.cleaned_items)}")
        print(f"❌ Errors encountered: {len(self.errors)}")
        
        if self.errors:
            print("\n⚠️ Some items couldn't be cleaned:")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"   - {error['action']}: {error['path']}")


if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent
    
    # Create cleaner instance
    cleaner = ProjectCleaner(project_root)
    
    # Run cleanup
    cleaner.run_full_cleanup()
    
    print("\n🎉 Project cleanup finished! Your project is now organized.")
