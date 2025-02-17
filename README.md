

API desarrollada con **FastAPI** para clasificar audios tomando como valores de entrada:
1. **Embeddings de Whisper** (modelo de OpenAI): Las representaciones vectoriales de la última capa del encoder de Whisper, rico en información acústica y semántica.

2. **Métricas de transcripción** (WER, sustituciones, inserciones, etc.): evaluación de la transcripción de lecturas de un texto.
---

## Estructura


- `main.py` → Contiene la API
- `config.py` → Configuración de modelos y dispositivos  
- `utils.py` → Funciones auxiliares para extracción de embeddings y métricas  
- `requirements.txt` → Librerías necesarias  
- `/models/` → Carpeta donde van los modelos `.pkl`  

---

## Instalación 

```bash
git https://github.com/joelqp/dl-dyslexia-detection.git
cd dl-dyslexia-detection

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

## Uso de la API

### **Endpoint "/predict" (POST)**
Permite enviar un archivo de audio junto con datos del usuario para clasificarlo.

#### **Parámetros del formulario:**

| Parámetro       | Tipo    | Descripción |
|----------------|--------|-------------|
| `audio`        | File   | Archivo de audio en formato `.wav`, `.mp3`, etc. |
| `edad`         | Int    | Edad del usuario |
| `curso`        | Int   | Curso de primaria |
| `genero`       | Int    | Género del usuario (M=0, F=1) |
| `lengua_materna` | Str  | Idioma materno del usuario (cat = 0, cast = 1, otro= -1) |
| `model_type`   | Str    | `"emb"` para embeddings o `"trans"` para transcripción |
| `use_augmented` | Bool  | `True` para usar el modelo con data augmentation |

#### **Ejemplo de una solicitud**
```bash
curl -X 'POST' 'http://localhost:8000/predict/' \
-F 'audio=@test_audio.wav' \
-F 'edad=12' \
-F 'curso="6to grado"' \
-F 'genero="M"' \
-F 'lengua_materna="español"' \
-F 'model_type="trans"' \
-F 'use_augmented=false'
```
#### Respuesta de ejemplo
```bash
{
    "model": "emb",
    "probability": 82.5,  # 82.5% de que el niño tenga dislexia
    "metadata": {
        "edad": "9",
        "curso": "4",
        "género": "1",
        "lengua_materna": "1"
    }
}

```
## Modelos Usados  
Este proyecto maneja **cuatro modelos diferentes**:

1. **`MODEL_EMB`**: Basado en **embeddings** de Whisper y un **SVM**.  
2. **`MODEL_EMB_AUG`**: Basado en **embeddings** de Whisper y **Random Forest** (el “modelo con data augmentation”).  
3. **`MODEL_TRANS_CTX`**: Basado en **transcripciones** (métricas tipo `substitutions`, `deletions`, etc.) y un **SVM**.  
4. **`MODEL_TRANS_CTX_AUG`**: Basado en **transcripciones** y **Random Forest** (también usando data augmentation).

---

## Explicación de los modelos

### Transcripción (trans):

Se usa la librería Whisper para transcribir el audio por chunks de 30s.
Luego se calculan las métricas de error (substitutions, deletions, insertions, etc.) con jiwer.
Además, se añaden datos como la duración del audio y la info de usuario (edad, curso, genero, lengua_materna).
Todo se normaliza con el scaler y se predice usando SVM o Random Forest.

### Embeddings (emb):

Se usan los últimos embeddings del encoder de Whisper (sacados del primer chunk del audio) y se pasa a un SVM o Random Forest para obtener la predicción.

---

- **Desarrollado por:** *Joel Marco Quiroga Poma*
- **Contacto:** [joelqp10@gmail.com]

