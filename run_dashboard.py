#!/usr/bin/env python
"""Quick setup and run script for the driver alertness detection dashboard."""

import subprocess
import sys
import os
import time
import webbrowser

def run_setup():
    """Setup and run the dashboard."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print("=" * 60)
    print("Driver Alertness Detection Dashboard - Setup & Run")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ required. Found {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Check Flask installation
    try:
        import flask
        print(f"✓ Flask {flask.__version__} found")
    except ImportError:
        print("✗ Flask not installed. Running: pip install -r requirements.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Check required modules
    try:
        import numpy
        import cv2
        print("✓ Core dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        sys.exit(1)
    
    # Set environment variables
    os.environ["FLASK_APP"] = "src.api.app"
    os.environ["FLASK_ENV"] = "development"
    os.environ["PYTHONPATH"] = project_root
    
    print("\n" + "=" * 60)
    print("Starting Flask development server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Start Flask server
    try:
        # Wait a moment before opening browser
        subprocess.Popen(
            [sys.executable, "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        # Give server time to start
        time.sleep(2)
        
        # Open browser
        print("\nOpening dashboard in browser...")
        webbrowser.open("http://localhost:5000")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nShutdown requested. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_setup()
