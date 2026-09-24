import shutil
import os

def nuke_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[DELETED] {path}")

if __name__ == "__main__":
    print("Sterilizing workspace...")
    
    nuke_directory("processed_darkened_data")
    nuke_directory("zero_dce/Zero-DCE_code/data/test_data")
    nuke_directory("zero_dce/Zero-DCE_code/data/result")
    
    for file in ["raw_results.json", "enhanced_results.json", "final_benchmark_curve.png"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"[DELETED] {file}")
            
    nuke_directory("raw_samples")
    os.makedirs("raw_samples", exist_ok=True)
    
    print("Workspace is clean. Drop new frames into 'raw_samples' before running the pipeline.")