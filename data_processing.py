import struct
import re
import os
import numpy as np
from scipy.fft import fft
from scipy.signal import stft, welch
from scipy.stats import kurtosis, skew

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
        print(f"An error occurred while processing file {filename}: {e}")

    return csi_data

def extract_features(csi_data):
    csi_array = np.array(csi_data)

    # Define window sizes and step sizes based on the scenario
    window_sizes = [100, 200, 400]  # 0.5秒，1秒，2秒
    step_sizes = [50, 100, 200]  # 0.25秒，0.5秒，1秒

    features = []

    for window_size in window_sizes:
        for step_size in step_sizes:
            for start in range(0, len(csi_array) - window_size + 1, step_size):
                window = csi_array[start:start + window_size]
                # Time domain features
                mean_val = np.mean(window)
                std_val = np.std(window)
                min_val = np.min(window)
                max_val = np.max(window)
                median_val = np.median(window)
                quartile_25 = np.percentile(window, 25)
                quartile_75 = np.percentile(window, 75)
                kurtosis_val = kurtosis(window)
                skewness_val = skew(window)
                zero_crossing_rate = ((window[:-1] * window[1:]) < 0).sum()
                energy = np.sum(window**2)

                features.extend([
                    mean_val, std_val, min_val, max_val, median_val,
                    quartile_25, quartile_75, kurtosis_val, skewness_val,
                    zero_crossing_rate, energy
                ])

                # Frequency domain features
                fft_values = np.abs(fft(window))
                fft_mean = np.mean(fft_values)
                fft_std = np.std(fft_values)
                fft_max = np.max(fft_values)
                fft_entropy = -np.sum((fft_values / np.sum(fft_values)) * np.log2(fft_values / np.sum(fft_values) + 1e-12))
                psd_values, psd_freqs = welch(window, nperseg=min(256, len(window)))
                psd_mean = np.mean(psd_values)
                psd_std = np.std(psd_values)
                psd_max = np.max(psd_values)
                psd_entropy = -np.sum((psd_values / np.sum(psd_values)) * np.log2(psd_values / np.sum(psd_values) + 1e-12))
                center_frequency = np.sum(psd_freqs * psd_values) / np.sum(psd_values)

                features.extend([
                    fft_mean, fft_std, fft_max, fft_entropy,
                    psd_mean, psd_std, psd_max, psd_entropy, center_frequency
                ])

                # Time-frequency domain features
                f, t, Zxx = stft(window, nperseg=min(256, len(window)))
                stft_magnitude = np.abs(Zxx)
                stft_mean = np.mean(stft_magnitude)
                stft_std = np.std(stft_magnitude)
                stft_max = np.max(stft_magnitude)

                features.extend([stft_mean, stft_std, stft_max])

    return features

def load_data(data_dir):
    fall_data = []
    no_fall_data = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith('txt'):
            label = filename.split('.')[0]
            file_path = os.path.join(data_dir, filename)
            print(f"Parsing file: {file_path}")
            csi_data = parse_file(file_path)
            
            if csi_data:
                print(f"Data found in file: {file_path}")
                features = extract_features(csi_data)
                if label == '1':
                    fall_data.append(features)
                    labels.append(1)
                elif label == '0':
                    no_fall_data.append(features)
                    labels.append(0)
            else:
                print(f"No CSI data found in file {filename}")
    
    return fall_data, no_fall_data, labels

def main(data_dir='data/'):
    fall_data, no_fall_data, labels = load_data(data_dir)
    all_data = fall_data + no_fall_data

    # Output the length of data lists for debugging
    print(f"Number of fall data samples: {len(fall_data)}")
    print(f"Number of no-fall data samples: {len(no_fall_data)}")
    print(f"Total number of samples: {len(all_data)}")

    # Ensure all feature vectors have the same length
    feature_length = len(all_data[0])
    print(f"Feature vector length: {feature_length}")

    for i, features in enumerate(all_data):
        if len(features) != feature_length:
            print(f"Inconsistent feature vector length at index {i}: {len(features)}")
            # Ensure the feature vector has the correct length by padding with zeros
            if len(features) < feature_length:
                features.extend([0] * (feature_length - len(features)))
            else:
                features = features[:feature_length]
            all_data[i] = features

    # Convert data to numpy array for easier manipulation
    all_data_np = np.array(all_data)
    labels_np = np.array(labels)

    # Output the shape of the data arrays for debugging
    print(f"Data array shape: {all_data_np.shape}")
    print(f"Labels array shape: {labels_np.shape}")

    # Save the data and labels to numpy files for future use
    np.save('csi_data.npy', all_data_np)
    np.save('csi_labels.npy', labels_np)
    
    print("Data and labels saved to 'csi_data.npy' and 'csi_labels.npy'.")

if __name__ == "__main__":
    main()
