import os
import pickle
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import logging

def extract_features(directory, output_file, max_folders=10):
    """
    Extract MFCC features from audio files in the specified directory and save them to a file.

    :param directory: Path to the directory containing audio files.
    :param output_file: Path to the output file where features will be saved.
    :param max_folders: Maximum number of folders to process.
    :return: Number of instances processed.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    instance_count = 0
    with open(output_file, "wb") as f:
        for i, folder in enumerate(sorted(os.listdir(directory)), 1):
            if i > max_folders:
                break

            logging.info(f"PROCESSING FOLDER: {folder}")

            for file in os.listdir(os.path.join(directory, folder)):
                file_path = os.path.join(directory, folder, file)
                
                if os.path.isfile(file_path) and file_path.endswith(".wav"):
                    try:
                        # Read wav file
                        (rate, sig) = wav.read(file_path)
                        
                        # Handle mono/stereo conversion
                        if sig.ndim > 1:
                            sig = sig[:, 0]

                        # Extract MFCC features
                        mfcc_feat = mfcc(
                            sig,
                            rate,
                            winlen=0.025,
                            winstep=0.01,
                            numcep=13,
                            nfilt=26,
                            nfft=1024,
                            appendEnergy=True,
                        )
                        
                        # Compute covariance and mean
                        covariance = np.cov(mfcc_feat.T)
                        mean_matrix = mfcc_feat.mean(0)
                        
                        # Feature structure: (mean_matrix, covariance, genre_label)
                        feature = (mean_matrix, covariance, i)
                        
                        # Save feature
                        pickle.dump(feature, f)
                        instance_count += 1
                    
                    except Exception as e:
                        logging.error(f"PROCESSING ERROR: {file_path}: {e}")
                        continue

    logging.info(f"TOTAL PROCESSED: {instance_count} instances")
    return instance_count