#!/usr/bin/env python3
"""
Setup script for the Content Scoring Tool
This script helps install the correct dependencies and resolve version conflicts.
"""

import subprocess
import sys

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_pytorch_version():
    """Check if PyTorch version is compatible"""
    try:
        import torch
        current_version = torch.__version__
        print(f"📋 Current PyTorch version: {current_version}")
        
        # Simple version check (not perfect but works for major versions)
        major_minor = current_version.split('.')[:2]
        if len(major_minor) >= 2:
            try:
                major = int(major_minor[0])
                minor = int(major_minor[1])
                if major > 2 or (major == 2 and minor >= 6):
                    print(f"✅ PyTorch {current_version} is compatible")
                    return True
                else:
                    print(f"❌ PyTorch {current_version} is too old. Required: >=2.6.0")
                    return False
            except ValueError:
                print(f"⚠️ Could not parse version {current_version}, proceeding anyway")
                return True
        else:
            print(f"⚠️ Could not parse version {current_version}, proceeding anyway")
            return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_requirements():
    """Install or upgrade requirements"""
    print("🔧 Installing/upgrading requirements...")
    
    # First, upgrade PyTorch to the latest version
    print("📦 Upgrading PyTorch...")
    success, stdout, stderr = run_command("pip install --upgrade torch>=2.6.0")
    
    if not success:
        print(f"❌ Failed to upgrade PyTorch: {stderr}")
        return False
    
    # Install other requirements
    print("📦 Installing other requirements...")
    success, stdout, stderr = run_command("pip install -r requirements_scoring_tool.txt")
    
    if not success:
        print(f"❌ Failed to install requirements: {stderr}")
        return False
    
    print("✅ All requirements installed successfully!")
    return True

def test_imports():
    """Test if all required modules can be imported"""
    modules_to_test = [
        'streamlit',
        'openai', 
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'textstat',
        'transformers',
        'torch',
        'plotly',
        'wordcloud'
    ]
    
    print("🧪 Testing imports...")
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0, failed_imports

def main():
    print("🚀 Content Scoring Tool Setup")
    print("=" * 40)
    
    # Check current PyTorch version
    pytorch_ok = check_pytorch_version()
    
    if not pytorch_ok:
        print("\n🔧 PyTorch needs to be upgraded...")
        if input("Do you want to upgrade PyTorch and install requirements? (y/n): ").lower() == 'y':
            if not install_requirements():
                print("❌ Setup failed!")
                sys.exit(1)
        else:
            print("⚠️ Setup cancelled. The tool may not work properly without the correct PyTorch version.")
            sys.exit(0)
    
    # Test imports
    print("\n" + "=" * 40)
    imports_ok, failed = test_imports()
    
    if imports_ok:
        print("\n✅ Setup completed successfully!")
        print("🚀 You can now run: streamlit run content_scoring_tool.py")
    else:
        print(f"\n❌ Setup incomplete. Failed imports: {', '.join(failed)}")
        print("💡 Try running: pip install -r requirements_scoring_tool.txt")

if __name__ == "__main__":
    main()
