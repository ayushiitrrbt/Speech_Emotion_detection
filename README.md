# 🎙 Speech Emotion Recognition Web App

This project is a web-based **Speech Emotion Recognition (SER)** system built with **Streamlit** and **TensorFlow**. It analyzes `.wav` audio files to predict the speaker's emotion using extracted audio features and a trained deep learning model.

---

## 🚀 Features

- 🎧 Upload a `.wav` audio file
- 📊 Audio feature extraction using `librosa`
- 🧠 Emotion classification using a pre-trained Keras model
- 🌈 Interactive UI with animations and styled output
- 📉 Displays probability distribution across emotion classes

---

## 🔍 How It Works

1. **Audio Upload**  
   Users upload a WAV file through the UI.

2. **Feature Extraction**  
   - MFCCs
   - Chroma
   - Mel Spectrogram
   - Spectral Contrast
   - Tonnetz
   - Spectral Flatness  
   All features are padded/truncated to 3 seconds.

3. **Preprocessing**  
   Scales the extracted features using `MinMaxScaler` and reduces dimensions using `PCA`.

4. **Prediction**  
   Passes the processed input to a trained Keras model for emotion prediction.

5. **Output**  
   Displays the predicted emotion and a bar chart of confidence scores.

---

## 📁 Folder Structure

