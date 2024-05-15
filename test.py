import numpy as np
import struct
import re
import joblib
import os

def hex_to_float(hex_str):
    try:
        if len(hex_str) == 8:
            return struct.unpack('<f', bytes.fromhex(hex_str))[0]
        else:
            return None
    except (ValueError, struct.error):
        return None

def parse_file(filename):
    hex_data_pattern = re.compile(r'^\s*0x[0-9a-fA-F]+:\s+([0-9a-fA-F ]+)')
    csi_data = []

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        for line in lines:
            match = hex_data_pattern.match(line)
            if match:
                hex_data = match.group(1).replace(' ', '')
                for i in range(0, len(hex_data), 8):
                    float_value = hex_to_float(hex_data[i:i+8])
                    if float_value is not None:
                        csi_data.append(float_value)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return csi_data

def extract_features(csi_data):
    if len(csi_data) == 0:
        return np.array([])

    window_size = 100
    step_size = 50
    features = []

    for start in range(0, len(csi_data) - window_size + 1, step_size):
        window = csi_data[start:start + window_size]
        mean = np.mean(window)
        std = np.std(window)
        max_val = np.max(window)
        min_val = np.min(window)
        features.append([mean, std, max_val, min_val])

    return np.array(features)

def preprocess_and_predict(filename, model_path='model.joblib', scaler_path='scaler.joblib'):
    csi_data = parse_file(filename)
    features = extract_features(csi_data)

    if features.size == 0:
        print("No valid CSI data extracted from the file.")
        return None

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)

    return int(np.round(np.mean(predictions)))

def main():
    data_dir = '.'
    for filename in os.listdir(data_dir):
        if filename.startswith('test') and filename.endswith('.txt'):
            print(f"Processing file: {filename}")
            result = preprocess_and_predict(os.path.join(data_dir, filename))
            if result is not None:
                print(f"Result for {filename}: {result}")

if __name__ == "__main__":
    main()
