#!/usr/bin/env python3
"""
Simple test script for retraining scheduler
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Test basic imports without heavy dependencies
    print("Testing basic imports...")

    # Test core classes without sklearn dependencies
    from retraining_scheduler import (
        ModelVersion, PerformanceMetrics, RetrainingResult,
        get_scheduler, start_scheduler, stop_scheduler
    )

    print("✓ Core classes imported successfully")

    # Test configuration loading
    from pathlib import Path
    import json

    config_file = Path("config/retraining_config.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print("✓ Configuration file loaded successfully")
        print(f"  - Models tracked: {len(config)} sections")
    else:
        print("⚠ Configuration file not found")

    # Test directory structure
    required_dirs = [
        "models",
        "models/versions",
        "data/logs",
        "config"
    ]

    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"⚠ Directory missing: {dir_path}")

    print("\n🎉 Retraining scheduler setup looks good!")
    print("\nNext steps:")
    print("1. Install APScheduler: pip install APScheduler")
    print("2. Start scheduler: python retraining_scheduler.py --start")
    print("3. Check status: python retraining_scheduler.py --status")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)