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

    def process_directory(self, directory: str, output_file: str, max_folders: int = 10):
        """
        Process all audio files in directory and save features
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
                        mean_matrix, covariance = self.extract_mfcc(file_path)
                        feature = (mean_matrix, covariance, i)
                        pickle.dump(feature, f)
                    except Exception as e:
                        print(f"ERROR PROCESSING AT {file}:\nERROR: {e}")
                        continue