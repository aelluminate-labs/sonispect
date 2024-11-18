import os
from pydub import AudioSegment
from pydub.utils import make_chunks


def convert_mp3_to_wav_and_cut(mp3_path, wav_path, chunk_length_ms=30000):
    # :: Ensure the output directory exists
    os.makedirs(wav_path, exist_ok=True)

    # :: Iterate over all MP3 files in the input directory
    for mp3_file in os.listdir(mp3_path):
        if mp3_file.endswith(".mp3"):
            try:
                # :: Construct the full path to the MP3 file
                mp3_file_full_path = os.path.join(mp3_path, mp3_file)

                # :: Load the MP3 file
                audio = AudioSegment.from_mp3(mp3_file_full_path)

                # :: Cut the audio into 30-second chunks
                chunks = make_chunks(audio, chunk_length_ms)

                # :: Export each chunk as a WAV file
                for i, chunk in enumerate(chunks):
                    chunk_name = (
                        f"{wav_path}/{os.path.splitext(mp3_file)[0]}_chunk_{i}.wav"
                    )
                    chunk.export(chunk_name, format="wav")
                    print(f"Exported {chunk_name}")
            except Exception as e:
                print(f"Failed to process {mp3_file}: {e}")


if __name__ == "__main__":
    # :: Specify the paths for the input MP3 directory and the output WAV directory
    mp3_path = "data/test/"
    wav_path = "data/preprocessed/converted/"

    # :: Check if the input directory exists
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"The input directory {mp3_path} does not exist.")

    # :: Convert the MP3 files to WAV
    convert_mp3_to_wav_and_cut(mp3_path, wav_path)
