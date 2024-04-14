import os
import librosa
import pandas as pd 

from datasets import Audio, Dataset


def create_dataset(data_dir: str = "/home/zhuldyz/Downloads/ISSAI_KazakhTTS2"):
    speaker_folders = os.listdir(data_dir)
    speaker_folders = speaker_folders[:-1]

    audio_paths = []
    audio_ids = []
    speaker_ids = []
    text_paths = []
    genders = []
    raw_texts = []
    normalized_texts = []

    for speaker_id in speaker_folders:
        directory_speaker = os.path.join(data_dir, speaker_id)
        audio_dir = os.path.join(directory_speaker, "Audio")
        audiofiles = os.listdir(audio_dir)
        transcripts_dir = os.path.join(directory_speaker, "Transcripts")
        transcripts = os.listdir(transcripts_dir)

        for i in range(len(audiofiles)):
            audio_path = os.path.join(audio_dir, audiofiles[i])
            text_path = os.path.join(transcripts_dir, transcripts[i])

            audio_ids.append(audiofiles[i][50:-4])
            audio_paths.append(audio_path)
            speaker_ids.append(speaker_id)
            text_paths.append(text_path)
            if speaker_id != 'M2' or speaker_id !='M1':
                genders.append('female')
            else:
                genders.append('male')

            with open(text_path, 'r') as file:
                raw_text = file.read()
                raw_texts.append(raw_text)
                normalized_texts.append(raw_text.lower())

    dataset = Dataset.from_dict({})
    
    # ["audio_id", "audio", "raw_text", "normalized_text", "gender", "speaker_id", "is_gold_transcript", "accent"]
    dataset.add_column('audio_id', audio_ids)
    dataset.add_column('audio', audio_paths)
    dataset.cast_column("audio", Audio())
    dataset.add_column('raw_text', raw_texts)
    dataset.add_column('normalized_text', normalized_texts)
    dataset.add_column('gender', genders)
    dataset.add_column('speaker_id', speaker_ids)

    return dataset


if __name__ == '__main__':
    dataset = create_dataset()
    for sample in dataset[:5]:
        print(sample)