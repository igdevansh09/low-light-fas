import subprocess
import shutil
import os

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[FATAL ERROR] Pipeline halted during: {command}")
        exit(1)

if __name__ == "__main__":
    print("=== STAGE 1: SYNTHETIC DEGRADATION ===")
    run_command("python batch_darken.py")

    print("\n=== STAGE 2: BASELINE BENCHMARK (RAW) ===")
    run_command("python benchmark.py processed_darkened_data raw_results.json")

    print("\n=== STAGE 3: DATA ROUTING ===")
    print("Copying degraded frames to Zero-DCE ingestion folder...")
    if os.path.exists("zero_dce/Zero-DCE_code/data/test_data"):
        shutil.rmtree("zero_dce/Zero-DCE_code/data/test_data")
    shutil.copytree("processed_darkened_data", "zero_dce/Zero-DCE_code/data/test_data")

    print("\n=== STAGE 4: ZERO-DCE ENHANCEMENT ===")
    run_command("python lowlight_test.py", cwd="zero_dce/Zero-DCE_code")

    print("\n=== STAGE 5: RECOVERED BENCHMARK (ENHANCED) ===")
    run_command("python benchmark.py zero_dce/Zero-DCE_code/data/result enhanced_results.json")
    
    print("\n=== STAGE 6: GENERATING FINAL GRAPH ===")
    run_command("python plot_results.py")