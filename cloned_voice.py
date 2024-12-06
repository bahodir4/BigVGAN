import os
import torch
import librosa
import soundfile as sf
from TTS.api import TTS
import bigvgan
from meldataset import get_mel_spectrogram


device = 'cuda' if torch.cuda.is_available() else 'cpu'


tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2") 


vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to(device)

# Ensure directories exist
os.makedirs("cloned_voice", exist_ok=True)

def clone_voice(real_audio_path, text, output_path):

    print(f"Cloning voice from: {real_audio_path}")
    wav_synthesized = tts.tts(
        text=text,
        speaker_wav=real_audio_path,
        language="en"
    )  # Directly get synthesized audio waveform

    
    wav_tensor = torch.FloatTensor(wav_synthesized).unsqueeze(0).to(device)  # [1, T_time]

    
    mel = get_mel_spectrogram(wav_tensor, vocoder.h).to(device)

    
    with torch.inference_mode():
        cloned_wav = vocoder(mel).squeeze().cpu().numpy()  # [T_time]

    
    sf.write(output_path, cloned_wav, int(vocoder.h.sampling_rate), subtype="PCM_16")
    print(f"Cloned voice saved to: {output_path}")


# Input parameters
real_audio_path = "./real_voice/Donald_Trump_voice.ogg"  # Path to real speaker's voice
text_input = "January 30, 1943: Admiral Erich Raeder resigns as Commander-in-Chief of the Kriegsmarine and is succeeded by Karl Dönitz due to growing dissatisfaction with Adolf Hitler following Germany's defeat in the Battle of the Barents Sea"
output_cloned_audio_path = "./cloned_voice/cloned_voice-6.wav"  # Path to save cloned audio

# Clone the voice
clone_voice(real_audio_path, text_input, output_cloned_audio_path)
