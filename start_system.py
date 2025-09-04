#!/usr/bin/env python3
"""
Smart Waste Detection System Launcher
Starts both backend and frontend with proper dependency checks
"""

import os
import sys
import subprocess
import time
import threading
import signal
from pathlib import Path

class SystemLauncher:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_process = None
        self.frontend_process = None
        self.processes = []
        
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        # Check Python dependencies
        try:
            import fastapi
            import uvicorn
            import ultralytics
            print("✅ Python dependencies OK")
        except ImportError as e:
            print(f"❌ Missing Python dependency: {e}")
            print("📦 Run: pip install -r requirements.txt")
            return False
        
        # Check Node.js and npm
        try:
            result = subprocess.run(['npm', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Node.js/npm OK")
            else:
                print("❌ npm not found")
                return False
        except FileNotFoundError:
            print("❌ Node.js/npm not installed")
            return False
        
        return True
    
    def setup_frontend(self):
        """Install frontend dependencies if needed"""
        frontend_dir = self.project_root / "waste-system" / "frontend"
        node_modules = frontend_dir / "node_modules"
        
        if not node_modules.exists():
            print("📦 Installing frontend dependencies...")
            try:
                subprocess.run(['npm', 'install'], 
                             cwd=frontend_dir, check=True)
                print("✅ Frontend dependencies installed")
            except subprocess.CalledProcessError:
                print("❌ Failed to install frontend dependencies")
                return False
        
        return True
    
    def start_backend(self):
        """Start the FastAPI backend"""
        print("🚀 Starting backend server...")
        
        backend_dir = self.project_root / "waste-system" / "backend"
        
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, "backend.py"],
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes.append(self.backend_process)
            
            # Monitor backend output in separate thread
            def monitor_backend():
                for line in self.backend_process.stdout:
                    print(f"[Backend] {line.strip()}")
            
            threading.Thread(target=monitor_backend, daemon=True).start()
            
            # Wait for backend to start
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                print("✅ Backend started successfully")
                return True
            else:
                print("❌ Backend failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the React frontend"""
        print("🎨 Starting frontend server...")
        
        frontend_dir = self.project_root / "waste-system" / "frontend"
        
        try:
            self.frontend_process = subprocess.Popen(
                ['npm', 'run', 'dev'],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes.append(self.frontend_process)
            
            # Monitor frontend output in separate thread
            def monitor_frontend():
                for line in self.frontend_process.stdout:
                    print(f"[Frontend] {line.strip()}")
            
            threading.Thread(target=monitor_frontend, daemon=True).start()
            
            # Wait for frontend to start
            time.sleep(5)
            
            if self.frontend_process.poll() is None:
                print("✅ Frontend started successfully")
                return True
            else:
                print("❌ Frontend failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting frontend: {e}")
            return False
    
    def cleanup(self):
        """Clean up processes on exit"""
        print("\n🧹 Shutting down services...")
        
        for process in self.processes:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    print(f"Warning: Error stopping process: {e}")
        
        print("✅ All services stopped")
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.cleanup()
        sys.exit(0)
    
    def run(self):
        """Main launcher function"""
        print("🗑️ Smart Waste Detection System Launcher")
        print("=" * 50)
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Check dependencies
        if not self.check_dependencies():
            print("\n❌ Dependency check failed. Please install missing dependencies.")
            return False
        
        # Setup frontend
        if not self.setup_frontend():
            print("\n❌ Frontend setup failed.")
            return False
        
        print("\n🚀 Starting services...")
        
        # Start backend
        if not self.start_backend():
            print("\n❌ Failed to start backend")
            self.cleanup()
            return False
        
        # Start frontend
        if not self.start_frontend():
            print("\n❌ Failed to start frontend")
            self.cleanup()
            return False
        
        print("\n" + "=" * 50)
        print("🎉 System is running!")
        print("📍 Frontend: http://localhost:5173")
        print("📍 Backend API: http://localhost:8000")
        print("📍 API Docs: http://localhost:8000/docs")
        print("=" * 50)
        print("💡 Press Ctrl+C to stop all services")
        
        # Keep the main process alive
        try:
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    print("❌ Backend process died")
                    break
                    
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("❌ Frontend process died")
                    break
                    
        except KeyboardInterrupt:
            pass
        
        self.cleanup()
        return True


def main():
    """Entry point"""
    launcher = SystemLauncher()
    success = launcher.run()
    
    if not success:
        print("\n❌ System failed to start properly")
        sys.exit(1)
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
