import os
import random

# Paths
metadata_path = './metadata.csv'
wavs_dir = 'wavs'
train_filelist_path = './train-full.txt'
val_filelist_path = './val-full.txt'

# Parameters
val_split_ratio = 0.1

# Read metadata
with open(metadata_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Shuffle data
random.shuffle(lines)

# Split data
val_size = int(len(lines) * val_split_ratio)
val_lines = lines[:val_size]
train_lines = lines[val_size:]



with open(train_filelist_path, 'w', encoding='utf-8') as f:
    for line in train_lines:
        parts = line.strip().split('|')
        audio_path =  f"{parts[0]}"
        transcription = parts[1]
        f.write(f"{audio_path}|{transcription}\n")

with open(val_filelist_path, 'w', encoding='utf-8') as f:
    for line in val_lines:
        parts = line.strip().split('|')
        audio_path =  f"{parts[0]}"
        transcription = parts[1]
        f.write(f"{audio_path}|{transcription}\n")
