#!/usr/bin/env python3

"""
Vibe Shopping Agent - Main Launcher
----------------------------------
This script launches the Vibe Shopping Agent application.
"""

import os
import sys
import subprocess

def main():
    print("╔══════════════════════════════════════╗")
    print("║        Vibe Shopping Agent           ║")
    print("╚══════════════════════════════════════╝")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(script_dir, "backend")
    
    # Check if backend directory exists
    if not os.path.exists(backend_dir):
        print("Error: Backend directory not found!")
        sys.exit(1)
    
    # Change to the backend directory
    os.chdir(backend_dir)
    
    # Check if .env file exists
    env_path = os.path.join(backend_dir, ".env")
    if not os.path.exists(env_path):
        print("Warning: .env file not found in backend directory.")
        print("You may need to configure API keys for full functionality.")
    
    print("\nStarting Vibe Shopping Agent server...")
    print("Open your browser at http://localhost:5001 once the server is running")
    print("Press Ctrl+C to stop the server\n")
    
    # Run the Flask app
    try:
        subprocess.run(["python", "app.py"])
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
