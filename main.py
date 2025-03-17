import streamlit as st
import whisper
import torch
import tempfile
import os
from pydub import AudioSegment
from io import BytesIO
from pydub.utils import which

# Vérifier la présence de FFmpeg et FFprobe
ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")

if ffmpeg_path and ffprobe_path:
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
    os.environ["FFPROBE_BINARY"] = ffprobe_path
else:
    raise FileNotFoundError("⚠️ FFmpeg et ffprobe ne sont pas installés ! Ajoutez-les dans packages.txt.")

# Vérifier si CUDA est disponible (accélération GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du modèle Whisper
@st.cache_resource
def load_model(model_size):
    return whisper.load_model(model_size, device=device)

# Fonction de transcription
def transcribe_audio(file, model_size="large"):
    model = load_model(model_size)

    # Convertir en format compatible si nécessaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        try:
            audio = AudioSegment.from_file(file)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(temp_wav.name, format="wav")
        except Exception as e:
            return f"Erreur lors de la conversion : {e}"
    
    # Lancer la transcription
    result = model.transcribe(temp_wav.name, language="fr")

    # Nettoyage
    os.remove(temp_wav.name)

    return result["text"]

# -------------------- INTERFACE STREAMLIT --------------------

st.set_page_config(page_title="Audio → Texte", page_icon="🎙️", layout="centered")

st.title("🎙️ Convertisseur Audio en Texte")
st.write("Téléchargez un fichier audio et obtenez sa transcription en texte.")

# Choix du modèle Whisper
model_size = st.selectbox("Choisissez un modèle, un modèle pérformant dit large veut dire plus de temps :", ["tiny", "base", "small", "medium", "large"], index=1)

# Upload fichier audio
uploaded_file = st.file_uploader("Téléchargez un fichier audio", type=["mp3", "wav", "flac", "m4a", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    if st.button("Convertir en texte"):
        with st.spinner("⏳ Transcription en cours..."):
            texte = transcribe_audio(uploaded_file, model_size)
            if texte:
                st.success("✅ Transcription terminée !")
                st.subheader("📝 Texte transcrit :")
                st.write(texte)

                # Ajouter un bouton pour télécharger le texte
                st.download_button("💾 Télécharger le texte", texte, file_name="transcription.txt", mime="text/plain")
            else:
                st.error("❌ Une erreur est survenue lors de la transcription.")

