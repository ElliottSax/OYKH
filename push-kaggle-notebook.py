#!/usr/bin/env python3
"""Push Kaggle notebook with proper UTF-8 encoding"""

import subprocess
import sys
import os

# Change to notebook directory
os.chdir('C:/projects/oykh-temp/kaggle-notebook-fixed')

try:
    # Run kaggle push with UTF-8 encoding
    result = subprocess.run(
        ['kaggle', 'kernels', 'push'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    print(f"\nReturn code: {result.returncode}")

    if result.returncode == 0:
        print("\n✅ Notebook pushed successfully!")
    else:
        print("\n❌ Push failed")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
