#!/usr/bin/env python
"""System verification and status check for the driver alertness detection project."""

import sys
import os
from pathlib import Path

def check_structure():
    """Verify project structure."""
    print("\n[1] Checking Project Structure")
    print("-" * 50)
    
    required_dirs = [
        "src/preprocessing",
        "src/api",
        "src/models",
        "src/utils",
        "templates",
        "static/css",
        "static/js",
        "tests"
    ]
    
    required_files = {
        "src/preprocessing/landmark_extractor.py": "Facial landmark extraction",
        "src/preprocessing/data_pipeline.py": "Dataset loading",
        "src/preprocessing/feature_engineering.py": "Feature aggregation",
        "src/api/app.py": "Flask backend with /predict endpoint",
        "templates/index.html": "Dashboard UI",
        "static/css/style.css": "Professional dark theme",
        "static/js/dashboard.js": "Dashboard controller",
        "tests/conftest.py": "Pytest configuration",
        "tests/test_landmark_extractor.py": "Landmark tests",
        "tests/test_data_pipeline.py": "Pipeline tests",
        "tests/test_feature_engineering.py": "Feature tests",
        "tests/test_api.py": "API tests",
        "tests/test_api_predict.py": "Endpoint tests",
        "README.md": "Documentation",
        "requirements.txt": "Dependencies",
        ".env.example": "Environment template"
    }
    
    all_ok = True
    
    # Check directories
    for d in required_dirs:
        path = Path(d)
        if path.exists():
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ - MISSING")
            all_ok = False
    
    print()
    
    # Check files
    for f, desc in required_files.items():
        path = Path(f)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {f}")
            print(f"      ({size} bytes) - {desc}")
        else:
            print(f"  ✗ {f} - MISSING")
            all_ok = False
    
    return all_ok


def check_dependencies():
    """Verify Python dependencies."""
    print("\n[2] Checking Python Dependencies")
    print("-" * 50)
    
    required = {
        'flask': 'Flask web framework',
        'numpy': 'Numerical computing',
        'cv2': 'Computer vision (OpenCV)',
        'pytest': 'Testing framework'
    }
    
    optional = {
        'mediapipe': 'Facial landmark detection',
        'torch': 'PyTorch (model training)',
        'tensorflow': 'TensorFlow (alternative)',
        'sklearn': 'Scikit-learn (ML utilities)'
    }
    
    print("\nRequired:")
    all_ok = True
    for name, desc in required.items():
        try:
            mod = __import__(name if name != 'cv2' else 'cv2')
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:15} {version:20} - {desc}")
        except ImportError:
            print(f"  ✗ {name:15} NOT INSTALLED - {desc}")
            all_ok = False
    
    print("\nOptional:")
    for name, desc in optional.items():
        try:
            mod = __import__(name if name != 'sklearn' else 'sklearn')
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:15} {version:20} - {desc}")
        except ImportError:
            print(f"  ⊘ {name:15} not installed (optional) - {desc}")
    
    return all_ok


def check_tests():
    """Run test suite and report results."""
    print("\n[3] Running Test Suite")
    print("-" * 50)
    
    try:
        import pytest
        result = pytest.main(['-q', '--tb=no'])
        if result == 0:
            print("  ✓ All tests passed")
            return True
        else:
            print(f"  ✗ Tests failed with code {result}")
            return False
    except Exception as e:
        print(f"  ✗ Failed to run tests: {e}")
        return False


def check_flask():
    """Verify Flask app initialization."""
    print("\n[4] Checking Flask App")
    print("-" * 50)
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.api.app import create_app
        app = create_app()
        
        print("  ✓ Flask app created successfully")
        
        # Check routes
        routes = []
        for rule in app.url_map.iter_rules():
            if rule.endpoint not in ['static']:
                routes.append(f"{rule.rule} -> {rule.endpoint}")
        
        print(f"\n  Routes registered ({len(routes)}):")
        for route in sorted(routes):
            print(f"    {route}")
        
        # Check endpoint availability
        endpoints = [r.endpoint for r in app.url_map.iter_rules() if r.endpoint != 'static']
        if 'index' in endpoints and 'predict' in endpoints:
            print("\n  ✓ Required endpoints present")
            return True
        else:
            print(f"\n  ✗ Missing endpoints. Found: {endpoints}")
            return False
            
    except Exception as e:
        print(f"  ✗ Failed to initialize Flask app: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_api_endpoint():
    """Test the /predict endpoint."""
    print("\n[5] Testing /predict Endpoint")
    print("-" * 50)
    
    try:
        from src.api.app import create_app, decode_image_from_base64
        import numpy as np
        import base64
        import cv2
        
        app = create_app()
        client = app.test_client()
        
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(encoded).decode('utf-8')
        
        # Test endpoint
        response = client.post('/predict', json={'data': b64})
        
        if response.status_code == 200:
            data = response.get_json()
            expected_keys = ['ear_left', 'ear_right', 'mar', 'pitch', 'yaw', 'roll', 'perclos', 'confidence']
            
            missing_keys = [k for k in expected_keys if k not in data]
            if missing_keys:
                print(f"  ✗ Response missing keys: {missing_keys}")
                return False
            
            print("  ✓ /predict endpoint working")
            print(f"\n  Sample response:")
            for key, value in data.items():
                print(f"    {key:12} = {value:.3f}")
            return True
        else:
            print(f"  ✗ Endpoint returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ✗ Failed to test endpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("DRIVER ALERTNESS DETECTION - SYSTEM VERIFICATION")
    print("=" * 70)
    
    results = {
        "Project Structure": check_structure(),
        "Dependencies": check_dependencies(),
        "Flask App": check_flask(),
        "API Endpoint": check_api_endpoint(),
        "Test Suite": check_tests()
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check:25} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - System ready for use!")
        print("\nTo start the dashboard:")
        print("  python run_dashboard.py")
        print("\nOr manually:")
        print("  FLASK_APP=src.api.app flask run --host=0.0.0.0 --port=5000")
        print("\nThen open browser: http://localhost:5000")
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt --break-system-packages")
    
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
