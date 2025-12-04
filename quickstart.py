#!/usr/bin/env python3
"""
Quick Start Script for Neural Networks Basics
Runs all examples and generates visualizations
"""

import os
import sys
import subprocess

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_script(script_path, description):
    """Run a Python script and display its output"""
    print(f"Running: {description}")
    print("-" * 70)
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {description}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Main function to run all examples"""
    print_header("Neural Networks Basics - Quick Start")
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(base_dir, "examples")
    viz_dir = os.path.join(base_dir, "visualizations")
    
    print("This script will run all neural network examples and generate visualizations.\n")
    
    # List of scripts to run
    scripts = [
        (os.path.join(examples_dir, "layerednn.py"), "Layered Neural Network"),
        (os.path.join(examples_dir, "newnn.py"), "Object-Oriented Neural Network"),
        (os.path.join(viz_dir, "nn_visualizer.py"), "Neural Network Visualizations"),
    ]
    
    results = []
    
    for script_path, description in scripts:
        print_header(description)
        if os.path.exists(script_path):
            success = run_script(script_path, description)
            results.append((description, success))
        else:
            print(f"✗ Script not found: {script_path}")
            results.append((description, False))
        print()
    
    # Print summary
    print_header("Summary")
    for description, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    print("\n" + "=" * 70)
    print("Check the 'assets' folder for generated visualizations!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
