
def get_file_paths_and_names(parent_dir, child_dir):
  import os
  import numpy as np
  list_of_files = os.listdir(os.path.join(os.getcwd(), parent_dir, child_dir))
  audio_paths = np.array([ os.path.join(os.getcwd(), parent_dir, child_dir, f_name) for f_name in list_of_files ])
  children_names = np.array(list(map(lambda x: x.split(' ')[0][:-1], np.array(list_of_files))))
  return audio_paths, children_names


def paths_labels_information():
    import numpy as np

    COURSES_DIR = "lecturas"
    COURSES = ["Segon primària", "Tercer primària", "Quart primària"]
    # DYSLEXICS_DIR = "Dislèctics - Okapis_cast"

    # dyslexic_paths, _ = get_file_paths_and_names("", DYSLEXICS_DIR)

    files_information = {}
    for course in COURSES:
        audio_paths, children_names = get_file_paths_and_names(COURSES_DIR, course)
        files_information[course] = {"audio_paths": audio_paths,  "children_names": children_names}

    # Check names are not duplicates
    for k, v in files_information.items():
            unique_elements, counts = np.unique(v["children_names"], return_counts=True)
            duplicates = unique_elements[counts > 1]
            if len(duplicates) > 0:
                raise ValueError("There are duplicate names")

    student_paths = files_information[COURSES[0]]["audio_paths"]
    for i in range(1, len(COURSES)):
        student_paths = np.concatenate((student_paths, files_information[COURSES[i]]["audio_paths"]), axis=0)


    # label_0 = len(student_paths) * [0]
    # label_1 = len(dyslexic_paths) * [1]

    all_paths = student_paths
    labels = np.array(len(student_paths) * [0])
    # labels = np.concatenate((label_0, label_1), axis=0)
    # all_paths = np.concatenate((student_paths, dyslexic_paths), axis=0)




    # etiquetar dislexia de los estudiantes de la escuela
    to_change = [24, 16, 32, 38, 50, 68, 52, 61, 62, 91, 57, 59, 84, 98, 129, 127]
    for v in to_change:
        labels[v] = 1

    # eliminar participantes mayores de 10 años
    # return all_paths[:143], labels[:143]
    return all_paths, labels, files_information


def segment_audios(audio_path, output_dir):
    import whisper
    from pydub import AudioSegment
    # Transcribir
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    audio = AudioSegment.from_file(audio_path)

    for i, segment in enumerate(result['segments']):
        start_time = int(segment['start'] * 1000)  # Convertir a milisegundos
        end_time = int(segment['end'] * 1000)
        phrase_audio = audio[start_time:end_time]

        output_path = f"{output_dir}/fragment_{i+1}.mp3"
        phrase_audio.export(output_path, format="mp3")
        print(f"Fragmento {i+1} guardado: {output_path}")
    return result

def transcriptions_and_embeddings(list_of_paths):
  import torch
  from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration
  import librosa
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.manifold import TSNE
  import umap

  processor = WhisperProcessor.from_pretrained("openai/whisper-base")
  model = WhisperModel.from_pretrained("openai/whisper-base")
  transcription_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  embeddings_list = []
  transcriptions_list = []
  for file in list_of_paths:
      speech_array, sampling_rate = librosa.load(file, sr=16000)
      inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt")

      with torch.no_grad():
          encoder_outputs = model.encoder(inputs.input_features)
          embeddings = encoder_outputs.last_hidden_state
          avg_embedding = embeddings.mean(dim=1).squeeze().numpy()
          embeddings_list.append(avg_embedding)

      with torch.no_grad():
          generated_ids = transcription_model.generate(inputs.input_features)
          transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
          transcriptions_list.append(transcription)

  embeddings_array = np.array(embeddings_list)
  transcriptions_array = np.array(transcriptions_list)
  return transcriptions_array, embeddings_array

#cambiarlo mas adelante por metricas mas robustas como inserciones y omisiones ?? creo que serviria para si pasa uno pero no lo otro seria raro ya que tienen que ser casi iguales ambas

def match_segments(gt, transcriptions_list):
    from difflib import SequenceMatcher
    gt_words = gt.split()
    current_index = 0 
    matched_segments = []

    for transcription in transcriptions_list:
        transcription_text = transcription[0].strip()
        best_match = ""
        best_ratio = 0
        best_end_index = current_index

        for end_index in range(current_index + 1, len(gt_words) + 1):
            candidate_segment = " ".join(gt_words[current_index:end_index])
            ratio = SequenceMatcher(None, candidate_segment, transcription_text).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate_segment
                best_end_index = end_index

        matched_segments.append({
            'transcription': transcription_text,
            'gt_segment': best_match,
            'similarity': best_ratio,
        })

        current_index = best_end_index

    return matched_segments
