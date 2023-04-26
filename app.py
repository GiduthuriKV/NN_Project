import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import librosa

def preprocess_audio(audio):
    y, sr = librosa.load(audio, mono=True, duration=5)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    feature_row = {
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'rolloff': np.mean(rolloff),
        'zero_crossing_rate': np.mean(zcr),
    }
    for i, c in enumerate(mfcc):
        feature_row[f'mfcc{i+1}'] = np.mean(c)

    return pd.DataFrame([feature_row])

# Load your trained Logistic Regression model
# Replace 'model.pkl' with the path to your saved model
model = pd.read_pickle('lgbm_classifier.pkl')
scaler = StandardScaler()

st.title('COVID-19 Cough Detection')
st.write('Upload a WAV file to determine if the cough is from a COVID-19 positive patient.')

uploaded_file = st.file_uploader('Choose a WAV file', type='wav')

if uploaded_file is not None:
    st.write('File uploaded successfully.')
    features = preprocess_audio(uploaded_file)
    features_scaled = scaler.fit_transform(features)
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.write('The cough is likely from a COVID-19 positive patient.')
    else:
        st.write('The cough is likely from a COVID-19 negative patient.')
else:
    st.write('No file uploaded.')
