#!/usr/bin/env python3
"""
Git Repository Cleaner for Smart Trash Detection System
Helps maintain a clean git repository with proper ignore patterns
"""

import os
import subprocess
import sys
from pathlib import Path

class GitCleaner:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.large_files = []
        self.ignored_tracked = []
        
    def check_git_status(self):
        """Check if we're in a git repository"""
        try:
            result = subprocess.run(['git', 'status'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            print("❌ Git not found. Please install Git.")
            return False
    
    def find_large_files(self, size_mb=10):
        """Find large files that shouldn't be in git"""
        print(f"🔍 Finding files larger than {size_mb}MB...")
        
        large_extensions = ['.pt', '.pth', '.h5', '.onnx', '.mp4', '.avi']
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                try:
                    size_mb_actual = file_path.stat().st_size / (1024 * 1024)
                    
                    if size_mb_actual > size_mb or file_path.suffix in large_extensions:
                        # Check if file is tracked
                        result = subprocess.run(
                            ['git', 'ls-files', str(file_path.relative_to(self.project_root))],
                            capture_output=True, text=True
                        )
                        
                        if result.stdout.strip():
                            self.large_files.append({
                                'path': file_path,
                                'size_mb': size_mb_actual,
                                'tracked': True
                            })
                            print(f"   📦 {file_path.relative_to(self.project_root)} ({size_mb_actual:.1f}MB) - TRACKED")
                        else:
                            self.large_files.append({
                                'path': file_path,
                                'size_mb': size_mb_actual,
                                'tracked': False
                            })
                            
                except (OSError, PermissionError):
                    continue
    
    def find_ignored_but_tracked(self):
        """Find files that are tracked but should be ignored"""
        print("🔍 Finding tracked files that should be ignored...")
        
        # Get all tracked files
        result = subprocess.run(['git', 'ls-files'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Failed to get tracked files")
            return
        
        tracked_files = result.stdout.strip().split('\n')
        
        for file_path in tracked_files:
            if not file_path:
                continue
                
            # Check if file should be ignored
            result = subprocess.run(
                ['git', 'check-ignore', file_path],
                capture_output=True, text=True
            )
            
            # If git check-ignore returns 0, the file is ignored
            if result.returncode == 0:
                self.ignored_tracked.append(file_path)
                print(f"   🚫 {file_path} - tracked but should be ignored")
    
    def untrack_files(self, files):
        """Remove files from git tracking without deleting them"""
        if not files:
            return True
            
        print("🗑️ Removing files from git tracking...")
        
        try:
            for file_path in files:
                result = subprocess.run(
                    ['git', 'rm', '--cached', str(file_path)],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    print(f"   ✅ Untracked: {file_path}")
                else:
                    print(f"   ❌ Failed to untrack: {file_path}")
                    
            return True
            
        except Exception as e:
            print(f"❌ Error untracking files: {e}")
            return False
    
    def add_gitattributes_lfs(self):
        """Add LFS tracking for large files"""
        print("📝 Checking Git LFS configuration...")
        
        # Check if Git LFS is available
        try:
            result = subprocess.run(['git', 'lfs', 'version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️ Git LFS not installed. Large files will be ignored instead.")
                return False
        except FileNotFoundError:
            print("⚠️ Git LFS not available. Large files will be ignored instead.")
            return False
        
        # Add LFS tracking for model files
        lfs_patterns = ['*.pt', '*.pth', '*.h5', '*.onnx']
        
        for pattern in lfs_patterns:
            result = subprocess.run(
                ['git', 'lfs', 'track', pattern],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print(f"   ✅ Added LFS tracking: {pattern}")
            else:
                print(f"   ⚠️ LFS tracking already exists: {pattern}")
        
        return True
    
    def optimize_repository(self):
        """Optimize git repository"""
        print("⚡ Optimizing git repository...")
        
        commands = [
            (['git', 'gc', '--aggressive'], "Garbage collection"),
            (['git', 'prune'], "Pruning unreachable objects"),
            (['git', 'repack', '-ad'], "Repacking objects")
        ]
        
        for cmd, description in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   ✅ {description}")
                else:
                    print(f"   ⚠️ {description} - {result.stderr.strip()}")
            except Exception as e:
                print(f"   ❌ {description} failed: {e}")
    
    def generate_report(self):
        """Generate cleaning report"""
        print("\n" + "="*50)
        print("📊 Git Repository Cleaning Report")
        print("="*50)
        
        print(f"📦 Large files found: {len(self.large_files)}")
        tracked_large = sum(1 for f in self.large_files if f['tracked'])
        print(f"   - Tracked: {tracked_large}")
        print(f"   - Untracked: {len(self.large_files) - tracked_large}")
        
        print(f"🚫 Files tracked but should be ignored: {len(self.ignored_tracked)}")
        
        if self.large_files:
            print("\n📦 Large files details:")
            for file_info in self.large_files[:10]:  # Show first 10
                status = "TRACKED" if file_info['tracked'] else "untracked"
                print(f"   - {file_info['path'].name} ({file_info['size_mb']:.1f}MB) - {status}")
        
        print("\n💡 Recommendations:")
        if tracked_large > 0:
            print("   - Remove large files from tracking")
            print("   - Consider using Git LFS for model files")
        
        if self.ignored_tracked:
            print("   - Update .gitignore and untrack ignored files")
        
        print("   - Run 'git gc' periodically to optimize repository")
        print("   - Use 'git lfs' for files larger than 100MB")
    
    def interactive_clean(self):
        """Interactive cleaning process"""
        print("🧹 Interactive Git Repository Cleaning")
        print("="*40)
        
        # Find large tracked files
        tracked_large = [f for f in self.large_files if f['tracked']]
        
        if tracked_large:
            print(f"\n📦 Found {len(tracked_large)} large tracked files:")
            for i, file_info in enumerate(tracked_large):
                print(f"   {i+1}. {file_info['path'].relative_to(self.project_root)} ({file_info['size_mb']:.1f}MB)")
            
            response = input("\n🤔 Remove these files from tracking? (y/N): ")
            if response.lower() in ['y', 'yes']:
                file_paths = [f['path'].relative_to(self.project_root) for f in tracked_large]
                self.untrack_files(file_paths)
        
        # Handle ignored but tracked files
        if self.ignored_tracked:
            print(f"\n🚫 Found {len(self.ignored_tracked)} files that are tracked but should be ignored:")
            for file_path in self.ignored_tracked[:5]:  # Show first 5
                print(f"   - {file_path}")
            
            if len(self.ignored_tracked) > 5:
                print(f"   ... and {len(self.ignored_tracked) - 5} more")
            
            response = input("\n🤔 Remove these files from tracking? (y/N): ")
            if response.lower() in ['y', 'yes']:
                self.untrack_files(self.ignored_tracked)
        
        # Optimize repository
        response = input("\n⚡ Optimize git repository? (y/N): ")
        if response.lower() in ['y', 'yes']:
            self.optimize_repository()
    
    def run_full_clean(self):
        """Run complete git cleaning process"""
        print("🚀 Starting git repository cleaning...")
        
        if not self.check_git_status():
            print("❌ Not in a git repository or git not available")
            return False
        
        # Find issues
        self.find_large_files()
        self.find_ignored_but_tracked()
        
        # Generate report
        self.generate_report()
        
        # Interactive cleaning
        self.interactive_clean()
        
        print("\n✨ Git repository cleaning completed!")
        return True


def main():
    """Entry point"""
    cleaner = GitCleaner()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Auto mode - just report, don't modify
        print("🔍 Running in analysis mode...")
        cleaner.check_git_status()
        cleaner.find_large_files()
        cleaner.find_ignored_but_tracked()
        cleaner.generate_report()
    else:
        # Interactive mode
        cleaner.run_full_clean()


if __name__ == "__main__":
    main()
