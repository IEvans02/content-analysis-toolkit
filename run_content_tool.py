#!/usr/bin/env python3
"""
🔬 Content Analysis Tool Launcher
==================================

Quick launcher for the content analysis tool with automatic dependency management.
"""

import subprocess
import sys
import os

def main():
    print("🔬 Starting Content Analysis Tool...")
    print("=" * 50)
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Activating virtual environment...")
        venv_activation = "source venv/bin/activate && "
    
    # Launch the streamlit app
    tool_path = "content_tools/content_analysis_tool.py"
    cmd = f"streamlit run {tool_path}"
    
    print(f"🚀 Launching: {cmd}")
    print("📍 The app will open at: http://localhost:8501")
    print("-" * 50)
    
    try:
        # Run the streamlit command
        if 'venv_activation' in locals():
            subprocess.run(f"{venv_activation}{cmd}", shell=True, check=True)
        else:
            subprocess.run(cmd.split(), check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching app: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())
