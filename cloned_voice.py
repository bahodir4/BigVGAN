"""import os
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
        language="ru"
    )  # Directly get synthesized audio waveform

    
    wav_tensor = torch.FloatTensor(wav_synthesized).unsqueeze(0).to(device)  # [1, T_time]

    
    mel = get_mel_spectrogram(wav_tensor, vocoder.h).to(device)

    
    with torch.inference_mode():
        cloned_wav = vocoder(mel).squeeze().cpu().numpy()  # [T_time]

    
    sf.write(output_path, cloned_wav, int(vocoder.h.sampling_rate), subtype="PCM_16")
    print(f"Cloned voice saved to: {output_path}")


# Input parameters
real_audio_path = "./real_voice/sokhib.wav"  # Path to real speaker's voice
text_input = "Tatariston muftiysi O‘zbekistondagi muftiy sayloviga mehmon sifatida taklif qilindi"
output_cloned_audio_path = "./cloned_voice/sokhib.wav"  # Path to save cloned audio

# Clone the voice
clone_voice(real_audio_path, text_input, output_cloned_audio_path)



"""
import os
import torch
import librosa
import soundfile as sf
import noisereduce as nr
from TTS.api import TTS
import bigvgan
from meldataset import get_mel_spectrogram

# Set device for computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Load BigVGAN vocoder
vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to(device)

# Ensure output directory exists
os.makedirs("cloned_voice", exist_ok=True)

# Function to transcribe audio to text with noise reduction
def transcribe_audio(audio_path):
    # Load the audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Step 1: Apply noise reduction
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr)
    
    # Save the noise-reduced audio temporarily for transcription
    temp_file = "temp_reduced_noise.wav"
    sf.write(temp_file, reduced_noise_audio, sr)
    
    # Use Whisper for transcription
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(temp_file)
    
    # Remove temporary file
    os.remove(temp_file)
    
    return result["text"]

# Function to clone voice with noise reduction
def clone_audio_to_voice(source_audio_path, target_voice_path, output_audio_path):
    print(f"Transcribing content from source audio: {source_audio_path}")
    
    # Step 1: Transcribe the source audio to text
    transcribed_text = transcribe_audio(source_audio_path)
    print(f"Transcribed Text: {transcribed_text}")

    # Step 2: Synthesize audio using the target speaker's voice
    print(f"Cloning audio into the target voice: {target_voice_path}")
    synthesized_wav = tts.tts(
        text=transcribed_text, 
        speaker_wav=target_voice_path,
        language="en"
    )

    # Step 3: Process the synthesized audio with the vocoder
    wav_tensor = torch.FloatTensor(synthesized_wav).unsqueeze(0).to(device)
    mel = get_mel_spectrogram(wav_tensor, vocoder.h).to(device)

    # Generate waveform with BigVGAN
    with torch.inference_mode():
        cloned_wav = vocoder(mel).squeeze().cpu().numpy()

    # Step 4: Save the cloned audio
    sf.write(output_audio_path, cloned_wav, int(vocoder.h.sampling_rate), subtype="PCM_16")
    print(f"Cloned voice saved to: {output_audio_path}")


# Input parameters
source_audio_path =  "./demo/examples/queen_24k.wav"  # Source audio (content to clone)
target_voice_audio_path = "./real_voice/sokhib.wav" # Your voice audio
output_cloned_audio_path = "./cloned_voice/cloned_audio-6.wav"  # Path to save cloned audio

# Clone the voice
clone_audio_to_voice(source_audio_path, target_voice_audio_path, output_cloned_audio_path)
