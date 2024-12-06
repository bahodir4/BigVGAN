import torch
import numpy as np
import librosa
from scipy.spatial.distance import euclidean

# Function to extract embeddings
def extract_embeddings(audio_path, feature_extractor, sr=22050):
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0) 

    
    embeddings = feature_extractor(wav_tensor)
    return embeddings.squeeze(0).cpu().detach().numpy()

def compute_similarity(real_embedding, generated_embedding):

    distance = euclidean(real_embedding, generated_embedding)
    return distance

real_audio_path = "./real_voice/bahodir.ogg"
generated_audio_path = "./cloned_voice/cloned_voice-5.wav"


def feature_extractor(audio_tensor):
    return audio_tensor.mean(dim=1, keepdim=True) 


real_embedding = extract_embeddings(real_audio_path, feature_extractor)
generated_embedding = extract_embeddings(generated_audio_path, feature_extractor)


print(f"Real embedding shape: {real_embedding.shape}")
print(f"Generated embedding shape: {generated_embedding.shape}")

similarity_score = compute_similarity(real_embedding, generated_embedding)
print(f"Similarity Score (Euclidean Distance): {similarity_score}")
