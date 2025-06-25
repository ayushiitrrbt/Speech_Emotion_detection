import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
from tensorflow.keras.models import load_model
import streamlit as st
import base64

# --- Page Background & Circular Logo UI ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://i.pinimg.com/736x/e8/73/47/e87347aeddaf058d75e59c202517d954.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# Animation: Show success using a Lottie animation (optional: you can change the URL)


st.title("🎙 Speech Emotion Recognition")

# --- Load model, scaler, and PCA ---
@st.cache_resource
def load_assets():
    model = load_model("emotion_keras_model_22113034.h5")
    with open("minmax_scaler_22113034.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("pca_transform_22113034.pkl", "rb") as f:
        pca = pickle.load(f)
    return model, scaler, pca

# --- Feature extraction ---
def extract_features(file_path, duration=3, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        desired_len = duration * sr
        if len(y) < desired_len:
            y = np.pad(y, (0, desired_len - len(y)))
        else:
            y = y[:desired_len]

        features = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))

        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features.extend(np.mean(mel, axis=1))

        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        features.extend(np.mean(contrast, axis=1))

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.extend(np.mean(tonnetz, axis=1))

        flatness = librosa.feature.spectral_flatness(y=y)
        features.append(np.mean(flatness))

        if len(features) != 194:
            st.error(f"❌ Feature length mismatch: expected 194, got {len(features)}")
            return None

        return np.array(features, dtype=np.float32).reshape(1, -1)

    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# --- File upload and inference ---
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')


    # Save to temp file so librosa can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    features = extract_features(temp_path)
    if features is not None:
        model, scaler, pca = load_assets()

        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        pred_probs = model.predict(features_pca)[0]
# Updated class mapping
        classes = {
            0: "angry",
            1: "calm",
            2: "disgust",
            3: "fearful",
            4: "happy",
            5: "neutral",
            6: "sad",
            7: "surprised"
        }

        # Get predicted class and display
        predicted_class = classes[np.argmax(pred_probs)]

        # Show result
                # Get predicted class
        predicted_class = classes[np.argmax(pred_probs)]

        # Styled result display
        st.markdown(f"""
            <div style="text-align: center; margin-top: 30px;">
                <h2 style="font-size: 32px; color: #ffffff;">Prediction:</h2>
                <p style="font-size: 48px; font-weight: bold; color: #00ffd5;">{predicted_class}</p>
            </div>
        """, unsafe_allow_html=True)


#
