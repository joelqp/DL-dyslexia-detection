import numpy as np
import librosa
import torch
import jiwer
from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration
from pydub import AudioSegment

def chunk_audio(signal, sr=16000, chunk_duration=30.0):
    """Divide la señal de audio en chunks de 'chunk_duration' segundos."""
    samples_per_chunk = int(chunk_duration * sr)
    return [signal[i:i+samples_per_chunk] for i in range(0, len(signal), samples_per_chunk)]

def extract_first_chunk_embedding(audio_data, processor, model, chunk_duration=30.0):
    """Extrae los embeddings de Whisper del primer chunk del audio."""
    chunks = chunk_audio(audio_data, chunk_duration=chunk_duration)
    if not chunks:
        return np.zeros(model.config.d_model)

    with torch.no_grad():
        inputs = processor(chunks[0], sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)
        encoder_outputs = model.encoder(input_features)
        embedding = encoder_outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    return embedding

def transcribe_audio(audio_data, model, processor, chunk_duration=30.0):
    """Transcribe un audio usando Whisper dividiéndolo en chunks."""
    chunks = chunk_audio(audio_data, chunk_duration=chunk_duration)
    full_transcription = ""

    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(input_features)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        full_transcription += " " + transcription.strip()
    return full_transcription.strip()

def compute_metrics(reference_text, hypothesis_text):
    """Calcula métricas de transcripción (WER, substitutions, etc.)."""
    measures = jiwer.compute_measures(reference_text, hypothesis_text)
    measures["cer"] = jiwer.cer(reference_text, hypothesis_text)
    measures["total_words_ref"] = len(reference_text.split())
    return measures
