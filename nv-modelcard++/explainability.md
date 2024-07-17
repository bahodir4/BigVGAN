Field                                                                                                  |  Response
:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------
Intended Application & Domain:                                                                 |   Generating waveform from mel spectrogram.
Model Type:                                                    |    Convolutional Neural Network (CNN)
Intended Users: | This model is intended for developers to synthesize and generate waveforms from the AI-generated mel spectrograms.
Output: | Audio Waveform
Describe how the model works: | Model generates audio waveform corresponding to the input mel spectrogram.
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | Not Applicable
Technical Limitations: | This may not perform well on synthetically-generated mel spectrograms that deviate significantly from the profile of mel spectrograms on which this was trained.
Verified to have met prescribed NVIDIA quality standards: |  Yes
Performance Metrics: | Perceptual Evaluation of Speech Quality (PESQ), Virtual Speech Quality Objective Listener (VISQOL), Multi-resolution STFT (MRSTFT), Mel cepstral distortion (MCD), Periodicity RMSE, Voice/Unvoiced F1 Score (V/UV F1)
Potential Known Risks: | This model may generate low-quality or distorted soundwaves.
Licensing: | https://github.com/NVIDIA/BigVGAN/blob/main/LICENSE