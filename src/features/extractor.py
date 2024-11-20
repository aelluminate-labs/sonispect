import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np
import os
import pickle
from typing import Tuple

class FeatureExtractor:
    def __init__(self, window_length: float = 0.020):
        self.window_length = window_length

    def extract_mfcc(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract MFCC features from an audio file.
        """
        rate, sig = wav.read(audio_path)
        mfcc_feat = mfcc(sig, rate, winlen=self.window_length, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        return mean_matrix, covariance

    def extract_nn_features(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract features suitable for Neural Networks (flattened mean and covariance).
        """
        rate, sig = wav.read(audio_path)
        mfcc_feat = mfcc(sig, rate, winlen=self.window_length, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        # Flatten and concatenate mean and covariance for NN features
        feature_vector = np.hstack([mean_matrix.flatten(), covariance.flatten()])
        return feature_vector

    def process_directory(self, directory: str, output_file: str, model_type: str, max_folders: int = 10):
        """
        Process all audio files in directory and save features dynamically based on model type.
        """
        with open(output_file, 'wb') as f:
            for i, folder in enumerate(os.listdir(directory), 1):
                if i > max_folders:
                    break

                folder_path = os.path.join(directory, folder)
                if not os.path.isdir(folder_path):
                    continue

                print(f"PROCESSING FOLDER: {folder}")
                
                for file in os.listdir(folder_path):
                    try:
                        file_path = os.path.join(folder_path, file)
                        label = i

                        # Dynamically select feature extraction method
                        if model_type == "knn" or model_type == "svm":
                            mean_matrix, covariance = self.extract_mfcc(file_path)
                            feature = (mean_matrix, covariance, label)
                        elif model_type == "nn":
                            feature_vector = self.extract_nn_features(file_path)
                            feature = (feature_vector, label)
                        else:
                            raise ValueError(f"Unsupported model type: {model_type}")

                        # Save the feature to output file
                        pickle.dump(feature, f)

                    except Exception as e:
                        print(f"ERROR PROCESSING FILE {file}: {e}")
                        continue
