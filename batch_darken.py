import os
import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_dataset(input_dir, output_root, gamma_levels=[0.8, 0.5, 0.3, 0.1]):
    for gamma in gamma_levels:
        os.makedirs(os.path.join(output_root, f"gamma_{gamma}"), exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found.")
        return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            if image is None: continue
                
            for gamma in gamma_levels:
                out_path = os.path.join(output_root, f"gamma_{gamma}", filename)
                cv2.imwrite(out_path, adjust_gamma(image, gamma=gamma))
                
    print("Batch darkening completed.")

if __name__ == "__main__":
    process_dataset("raw_samples", "processed_darkened_data")