#!/usr/bin/env python3
"""
Dependency checker for Wan2.2-S2V-14B

Scans Python files for imports and checks if they're installed.
Run locally before building Docker to catch missing deps early.

Usage: python check_deps.py [path_to_wan_repo]
"""

import ast
import sys
import subprocess
from pathlib import Path

# Core imports that must work (from wan module)
REQUIRED_IMPORTS = [
    "wan",
    "wan.speech2video",
    "decord",
    "torch",
    "torchvision", 
    "torchaudio",
    "diffusers",
    "transformers",
    "accelerate",
    "PIL",
    "numpy",
    "cv2",
    "imageio",
    "safetensors",
    "tqdm",
    "easydict",
    "ftfy",
]

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name.split('.')[0])
        return True, None
    except ImportError as e:
        return False, str(e)

def extract_imports_from_file(filepath):
    """Extract all imports from a Python file."""
    imports = set()
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except:
        pass
    return imports

def main():
    print("=" * 60)
    print("Wan2.2-S2V Dependency Checker")
    print("=" * 60)
    
    # Check required imports
    print("\nChecking required imports...")
    missing = []
    for module in REQUIRED_IMPORTS:
        ok, err = check_import(module)
        status = "OK" if ok else f"MISSING ({err})"
        print(f"  {module}: {status}")
        if not ok:
            missing.append(module)
    
    # Scan wan directory if provided
    if len(sys.argv) > 1:
        wan_path = Path(sys.argv[1])
        if wan_path.exists():
            print(f"\nScanning {wan_path} for imports...")
            all_imports = set()
            for py_file in wan_path.rglob("*.py"):
                all_imports.update(extract_imports_from_file(py_file))
            
            # Filter to third-party only (not stdlib, not local)
            stdlib = {'os', 'sys', 'math', 'random', 'logging', 'warnings', 
                     'datetime', 'contextlib', 'copy', 'functools', 'gc',
                     'types', 'argparse', 'json', 'time', 'typing', 'abc',
                     'collections', 'itertools', 'pathlib', 'tempfile'}
            third_party = all_imports - stdlib - {'wan'}
            
            print(f"\nThird-party imports found: {sorted(third_party)}")
    
    # Summary
    print("\n" + "=" * 60)
    if missing:
        print(f"FAILED: {len(missing)} missing dependencies")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    else:
        print("PASSED: All dependencies available")
        sys.exit(0)

if __name__ == "__main__":
    main()
