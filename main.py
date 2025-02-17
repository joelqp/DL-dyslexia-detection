from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import numpy as np
import librosa
import torch
import pickle
from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration
from config import *
from utils import extract_first_chunk_embedding, transcribe_audio, compute_metrics
from pydub import AudioSegment

app = FastAPI()

with open(MODEL_EMB, 'rb') as f:
    model_svm_emb = pickle.load(f)

with open(MODEL_EMB_AUG, 'rb') as f:
    model_rf_emb = pickle.load(f)

with open(MODEL_TRANS_CTX, 'rb') as f:
    model_svm_trans = pickle.load(f)

with open(MODEL_TRANS_CTX_AUG, 'rb') as f:
    model_rf_trans = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="es", task="transcribe")
whisper_model = WhisperModel.from_pretrained("openai/whisper-base").to(device).eval()
whisper_transcriber = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device).eval()

@app.post("/predict/")
async def predict(
    audio: UploadFile = File(...),
    edad: int = Form(...),
    curso: int = Form(...),
    genero: int = Form(...),
    lengua_materna: int = Form(...),
    model_type: str = Form("trans"),
    use_augmented: bool = Form(False)
):
    try:
        audio_data, sr = librosa.load(audio.file, sr=16000)
        duration = len(AudioSegment.from_file(audio.file)) / 1000

        if model_type == "emb":
            embedding = extract_first_chunk_embedding(audio_data, processor, whisper_model)
            input_data = embedding.reshape(1, -1)
            model = model_rf_emb if use_augmented else model_svm_emb

        elif model_type == "trans":
            transcription = transcribe_audio(audio_data, whisper_transcriber, processor)
            metrics = compute_metrics(REFERENCE_TEXT, transcription)


            input_data = np.array([[
                metrics["substitutions"],
                metrics["deletions"],
                metrics["insertions"],
                duration,
                edad,
                curso,
                genero,
                lengua_materna
            ]])

            input_data = scaler.transform(input_data)
            model = model_rf_trans if use_augmented else model_svm_trans

        else:
            raise HTTPException(status_code=400, detail="Modelo no válido. Usa 'emb' o 'trans'.")

        # Hacer predicción
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        return {
            "prediction": int(prediction[0]),
            "probabilities": probabilities.tolist(),
            "metadata": {
                "edad": edad,
                "curso": curso,
                "genero": genero,
                "lengua_materna": lengua_materna
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
