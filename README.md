# Sonispect

Classify 10 music genres. Extract Mel Frequency Cepstral Coefficients for feature extraction. This project involves segmenting audio signals, identifying frequencies, separating linguistic content from noise, and applying discrete cosine transform to focus on informative frequencies. 

## About the Dataset

The GTZAN genre collection dataset was collected in 2000-2001. It consists of 1000 audio files each having 30 seconds duration. There are 10 classes ( 10 music genres) each containing 100 audio tracks. Each track is in .wav format. It contains audio files of the following 10 genres:

- Blues
- Classical 
- Country 
- Disco 
- Hiphop 
- Jazz
- Metal 
- Pop 
- Reggae 
- Rock

### Content

- **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
- **images original** - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
- **2 CSV files** - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.

## Model Performance

| Model | Version | Accuracy | Remarks |
| --- | --- | --- | --- |
| KNN | v1 | 🔺67.13% | - |
| SVM (multi-class) | v1 | 🔻63.43% | - |