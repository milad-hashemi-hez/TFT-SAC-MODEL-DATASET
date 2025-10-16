
#!/usr/bin/env python3
"""
CACHE CLEANER MODULE
Independent utility to clear all cache before training
Prevents data contamination and ensures clean training runs
"""

import os
import shutil
import glob

def clear_python_cache():
    """Remove Python cache files and directories"""
    cache_dirs = ["__pycache__", ".pytest_cache", ".mypy_cache"]
    cache_files = ["*.pyc", "*.pyo"]
    
    print("🧹 Cleaning Python cache...")
    
    # Remove cache directories
    for dir_name in cache_dirs:
        for root, dirs, files in os.walk('.'):
            if dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f"✅ Removed: {dir_path}")
    
    # Remove cache files
    for pattern in cache_files:
        for file_path in glob.glob(f"**/{pattern}", recursive=True):
            os.remove(file_path)
            print(f"✅ Removed: {file_path}")

def clear_training_artifacts():
    """Remove training-specific files and models"""
    artifacts = [
        "best_tft_sac_model.pth",
        "final_tft_sac_model.pth",
        "training_logs",
        "runs",  # TensorBoard
        "logs"
    ]
    
    print("🧹 Cleaning training artifacts...")
    
    for item in artifacts:
        # Handle file patterns
        if '*' in item:
            for file_path in glob.glob(item):
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"✅ Removed: {file_path}")
                    else:
                        shutil.rmtree(file_path)
                        print(f"✅ Removed: {file_path}")
        # Handle specific items
        elif os.path.exists(item):
            if os.path.isfile(item):
                os.remove(item)
                print(f"✅ Removed: {item}")
            else:
                shutil.rmtree(item)
                print(f"✅ Removed: {item}")

def clear_model_checkpoints():
    """Remove all model checkpoint files"""
    print("🧹 Cleaning model checkpoints...")
    
    checkpoint_patterns = [
        "tft_sac_model_ep*.pth",
        "*_model.pth",
        "checkpoint_*.pth"
    ]
    
    for pattern in checkpoint_patterns:
        for file_path in glob.glob(pattern):
            os.remove(file_path)
            print(f"✅ Removed: {file_path}")

def clear_all_cache():
    """
    MAIN FUNCTION: Clear all cache and artifacts
    Call this before every training session
    """
    print("🚀 STARTING CACHE CLEANUP...")
    print("=" * 50)
    
    clear_python_cache()
    clear_training_artifacts()
    clear_model_checkpoints()
    
    print("=" * 50)
    print("🎯 CACHE CLEANUP COMPLETED!")
    print("💡 Your project is now ready for clean training")

if __name__ == "__main__":
    # Run when executed directly
    clear_all_cache()