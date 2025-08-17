#!/usr/bin/env python3
"""
Simple script to install file processing dependencies for the Content Scoring Tool
"""

import subprocess
import sys

def install_package(package_name, import_name=None):
    """Install a package and test if it can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False

def main():
    print("🚀 Installing File Processing Dependencies")
    print("=" * 50)
    
    packages = [
        ("python-docx", "docx"),
        ("PyPDF2", "PyPDF2"),
    ]
    
    success_count = 0
    
    for package_name, import_name in packages:
        if install_package(package_name, import_name):
            success_count += 1
        print()
    
    print("=" * 50)
    if success_count == len(packages):
        print("🎉 All dependencies installed successfully!")
        print("🚀 You can now run the content scoring tool with full file support:")
        print("   streamlit run content_scoring_tool_lite.py")
    else:
        print(f"⚠️ {len(packages) - success_count} packages failed to install")
        print("💡 Try installing manually:")
        for package_name, _ in packages:
            print(f"   pip install {package_name}")

if __name__ == "__main__":
    main()
