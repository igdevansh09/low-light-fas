import json
import matplotlib.pyplot as plt
import os

def load_results(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Did the benchmark run?")
        return {}
    with open(filepath, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    raw_data = load_results("raw_results.json")
    enhanced_data = load_results("enhanced_results.json")

    gamma_keys = ["gamma_0.8", "gamma_0.5", "gamma_0.3", "gamma_0.1"]
    gamma_labels = [0.8, 0.5, 0.3, 0.1]

    raw_confidence = [raw_data.get(k, 0) for k in gamma_keys]
    enhanced_confidence = [enhanced_data.get(k, 0) for k in gamma_keys]

    plt.figure(figsize=(10, 6))
    plt.plot(gamma_labels, raw_confidence, marker='o', linestyle='-', color='red', label='Raw Darkened (Degraded)')
    plt.plot(gamma_labels, enhanced_confidence, marker='s', linestyle='--', color='blue', label='Zero-DCE Enhanced (Recovered)')

    plt.title('Robustness of Face Anti-Spoofing: Accuracy vs Illumination')
    plt.xlabel('Illumination Level (Gamma)')
    plt.ylabel('Average Model Confidence Score')
    plt.xticks(gamma_labels, ['Gamma 0.8\n(Dim)', 'Gamma 0.5', 'Gamma 0.3', 'Gamma 0.1\n(Near Dark)'])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('final_benchmark_curve.png')
    print("\n[SUCCESS] Benchmark curve generated and saved as final_benchmark_curve.png")