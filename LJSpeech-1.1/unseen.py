import os
import csv
import shutil

def create_unseen_data_from_csv(csv_file, wavs_dir, unseen_clean_dir, unseen_other_dir, dev_clean_file, dev_other_file, num_files=100):
    """
    Processes the LJSpeech metadata.csv to create unseen datasets for evaluation.

    Parameters:
    - csv_file: Path to the metadata.csv file.
    - wavs_dir: Directory containing the original WAV audio files.
    - unseen_clean_dir: Directory to store the 'clean' subset of unseen audio files.
    - unseen_other_dir: Directory to store the 'other' subset of unseen audio files.
    - dev_clean_file: Output file listing the 'clean' unseen audio files and their transcriptions.
    - dev_other_file: Output file listing the 'other' unseen audio files and their transcriptions.
    - num_files: Number of files to include in the 'clean' subset. The remainder will go to the 'other' subset.
    """
    # Ensure output directories exist
    os.makedirs(unseen_clean_dir, exist_ok=True)
    os.makedirs(unseen_other_dir, exist_ok=True)

    dev_clean_list = []
    dev_other_list = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            # Check if the row has the expected number of columns
            if len(row) < 3:
                print(f"Warning: Row has fewer than 3 columns: {row}")
                continue

            file_id = row[0].strip()
            normalized_text = row[2].strip()  # Normalized transcription

            wav_path = os.path.join(wavs_dir, f"{file_id}.wav")
            if not os.path.exists(wav_path):
                print(f"Warning: {wav_path} not found!")
                continue

            # Determine destination directory and list
            if len(dev_clean_list) < num_files:
                dest_dir = unseen_clean_dir
                dev_clean_list.append(f"{file_id}|{normalized_text}")
            else:
                dest_dir = unseen_other_dir
                dev_other_list.append(f"{file_id}|{normalized_text}")

            # Copy the WAV file to the destination directory
            shutil.copy(wav_path, os.path.join(dest_dir, f"{file_id}.wav"))

    # Write the 'clean' subset filelist
    with open(dev_clean_file, 'w', encoding='utf-8') as f:
        for item in dev_clean_list:
            f.write(f"{item}\n")

    # Write the 'other' subset filelist
    with open(dev_other_file, 'w', encoding='utf-8') as f:
        for item in dev_other_list:
            f.write(f"{item}\n")

    print(f"Unseen data preparation complete. {len(dev_clean_list)} files in {unseen_clean_dir}, {len(dev_other_list)} files in {unseen_other_dir}.")

# Parameters
csv_file = './metadata.csv'
wavs_dir = './wavs'
unseen_clean_dir = './unseen_clean'
unseen_other_dir = './unseen_other'
dev_clean_file = 'dev-clean.txt'
dev_other_file = 'dev-other.txt'

# Execute the function
create_unseen_data_from_csv(
    csv_file=csv_file,
    wavs_dir=wavs_dir,
    unseen_clean_dir=unseen_clean_dir,
    unseen_other_dir=unseen_other_dir,
    dev_clean_file=dev_clean_file,
    dev_other_file=dev_other_file,
    num_files=100  # Number of files for the clean subset
)
