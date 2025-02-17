import os
from pydub import AudioSegment

def segment_audio_uniform(audio_path, output_dir, segment_ms=3000):
    """
    Corta un archivo de audio en trozos de segment_ms milisegundos y los guarda en output_dir.
    Retorna una lista con las rutas de los trozos generados.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    audio = AudioSegment.from_file(audio_path)
    audio_length_ms = len(audio)
    
    chunk_paths = []
    start = 0
    idx = 0
    
    while start < audio_length_ms:
        end = start + segment_ms
        if end > audio_length_ms:
            end = audio_length_ms
        
        chunk = audio[start:end]
        chunk_path = os.path.join(output_dir, f"chunk_{idx}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
        
        start += segment_ms
        idx += 1
    
    return chunk_paths

def chunk_to_melspectrogram(chunk_path, 
                            sr=16000, 
                            n_mels=64, 
                            hop_length=512, 
                            n_fft=1024,
                            max_time_frames=128):
    """
    Convierte un trozo de audio en un mel-espectrograma logarítmico y
    lo pad/trunca a max_time_frames en el eje temporal.
    
    Retorna un array numpy (n_mels, max_time_frames, 1).
    """
    signal, sr = librosa.load(chunk_path, sr=sr)
    
    S = librosa.feature.melspectrogram(y=signal, sr=sr,
                                       n_mels=n_mels,
                                       hop_length=hop_length,
                                       n_fft=n_fft)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    current_time_frames = log_S.shape[1]
    
    if current_time_frames < max_time_frames:
        pad_width = max_time_frames - current_time_frames
        log_S = np.pad(log_S, ((0,0), (0, pad_width)), mode='constant')
    elif current_time_frames > max_time_frames:
        log_S = log_S[:, :max_time_frames]
    
    log_S = np.expand_dims(log_S, axis=-1)
    
    return log_S

import random
import tensorflow as tf
import numpy as np
import os
import numpy as np
from pydub import AudioSegment
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

def create_dataset(audio_paths, labels, idx_list, 
                   segment_ms=3000, 
                   sr=16000, 
                   n_mels=64, 
                   hop_length=512, 
                   n_fft=1024,
                   max_time_frames=128  # O el número que decidas
                   ):
    """
    Crea un dataset de (spectrogram, label) a partir de una lista
    de (audio_path, label).
    - 'label' es 0/1 (ejemplo binario dislexia/no dislexia)
    - Cada trozo de un mismo audio mantiene la misma etiqueta.
    """


    X = []
    Y = []
    
    for i in idx_list:
        audio_path = audio_paths[i]
        label = labels[i]
        
        # Creamos un directorio temporal para los trozos de este audio
        # Podrías usar un único dir e ir limpiando, o algo más sofisticado
        output_dir = "temp_chunks"
        
        # Segmentar uniformemente en trozos de segment_ms
        chunk_paths = segment_audio_uniform(
            audio_path=audio_path,
            output_dir=output_dir,
            segment_ms=segment_ms
        )
        
        for ch_path in chunk_paths:
            spec = chunk_to_melspectrogram(
                ch_path, sr=sr, n_mels=n_mels, 
                hop_length=hop_length, n_fft=n_fft,
                max_time_frames=max_time_frames
            )
            X.append(spec)
            Y.append(label)
    
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"Created dataset from {len(idx_list)} audios -> total chunks: {len(X)}")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    
    return X, Y



def build_cnn(input_shape):
    from tensorflow.keras import models, layers, regularizers
    """
    Construye una CNN mejorada para espectrogramas.
    input_shape = (n_mels, time_frames, 1)
    """
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())  # Normalización para estabilizar el aprendizaje
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))  # Regularización

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model



# def build_cnn(input_shape):
#     """
#     Construye una CNN sencilla para espectrogramas:
#     input_shape = (n_mels, time_frames, 1)
#     """
#     model = models.Sequential()
    
#     model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
#     model.add(layers.MaxPooling2D((2,2)))
    
#     model.add(layers.Conv2D(64, (3,3), activation='relu'))
#     model.add(layers.MaxPooling2D((2,2)))
    
#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='relu'))
    
#     # Capa de salida binaria para dislexia o no
#     model.add(layers.Dense(1, activation='sigmoid'))
    
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
    
#   return model


