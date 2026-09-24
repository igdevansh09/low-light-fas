import os
import cv2
import numpy as np
import sys
import warnings
import json

sys.path.append(os.path.abspath("antispoof_model"))
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

def evaluate_lighting_benchmark(darkened_root_dir, output_json_path, model_dir="antispoof_model/resources/anti_spoof_models", device_id=0):
    print("Initializing Model...")
    
    original_dir = os.getcwd()
    os.chdir("antispoof_model")
    model_test = AntiSpoofPredict(device_id)
    os.chdir(original_dir)
    
    image_cropper = CropImage()
    results_dict = {}
    
    for gamma_folder in sorted(os.listdir(darkened_root_dir), reverse=True):
        gamma_path = os.path.join(darkened_root_dir, gamma_folder)
        if not os.path.isdir(gamma_path): continue
            
        print(f"\n--- Evaluating Condition: {gamma_folder} ---")
        total_images, successful_predictions, cumulative_confidence = 0, 0, 0.0
        
        for filename in os.listdir(gamma_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(os.path.join(gamma_path, filename))
                if image is None: continue
                    
                try:
                    image_bbox = model_test.get_bbox(image)
                    if sum(image_bbox) == 0: continue 
                        
                    prediction = np.zeros((1, 3))
                    for model_name in os.listdir(model_dir):
                        if not model_name.endswith('.pth'): continue
                        h_input, w_input, model_type, scale = parse_model_name(model_name)
                        param = {"org_img": image, "bbox": image_bbox, "scale": scale, "out_w": w_input, "out_h": h_input, "crop": True}
                        if scale is None: param["crop"] = False
                            
                        img = image_cropper.crop(**param)
                        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                    
                    label = np.argmax(prediction)
                    score = prediction[0][label] / 2.0
                    
                    cumulative_confidence += score
                    successful_predictions += 1
                    total_images += 1
                except Exception:
                    continue
        
        avg_conf = (cumulative_confidence / successful_predictions) if successful_predictions > 0 else 0.0
        print(f"Processed: {total_images} | Average Confidence: {avg_conf:.4f}")
        results_dict[gamma_folder] = avg_conf

    with open(output_json_path, "w") as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    eval_dir = sys.argv[1] if len(sys.argv) > 1 else "processed_darkened_data"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "raw_results.json"
    evaluate_lighting_benchmark(eval_dir, out_file)