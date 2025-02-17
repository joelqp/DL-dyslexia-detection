import torch


MODEL_EMB = "models/model_audio.pkl"
MODEL_EMB_AUG = "models/model_audio_aug.pkl"
MODEL_TRANS_CTX = "models/model_ctx_trans.pkl"
MODEL_TRANS_CTX_AUG = "models/model_ctx_trans_aug.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALER_PATH = "scaler/scaler.pkl"

REFERENCE_TEXT = """Los okapis son animales mamíferos que viven en las selvas de África. Son casi tan grandes como las jirafas y tienen rayas como las cebras. Tienen un hocico fuerte y con su lengua pueden limpiarse hasta las orejas."""
