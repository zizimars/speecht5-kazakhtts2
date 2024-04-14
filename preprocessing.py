import code
import os
import librosa
import pandas as pd
from transformers import SpeechT5Processor
from datasets import Dataset, Audio


def create_dataset():
    base_directory_path = "/home/zhuldyz/Downloads/ISSAI_KazakhTTS2"
    speaker_folders = os.listdir(base_directory_path)
    speaker_folders = speaker_folders[:-1]

    audio_paths = []
    speaker_ids = []
    text_paths = []

    for folder in speaker_folders:
        directory_speaker = os.path.join(base_directory_path, folder)
        audio_dir = os.path.join(directory_speaker, "Audio")
        audiofiles = os.listdir(audio_dir)
        transcripts_dir = os.path.join(directory_speaker, "Transcripts")
        transcripts = os.listdir(transcripts_dir)

        for i in range(len(audiofiles)):
            audio_path = os.path.join(audio_dir, audiofiles[i])
            text_path = os.path.join(transcripts_dir, transcripts[i])

            audio_paths.append(audio_path)
            speaker_ids.append(folder)
            text_paths.append(text_path)

    with open("/home/zhuldyz/Downloads/KazSpeechT5/audio_paths.txt", 'w') as file:
        for line in audio_paths:
            file.write(f"{line}\n")

    with open("/home/zhuldyz/Downloads/KazSpeechT5/speakers_id.txt", 'w') as file:
        for line in speaker_ids:
            file.write(f"{line}\n")

    with open("/home/zhuldyz/Downloads/KazSpeechT5/transcripts_paths.txt", 'w') as file:
        for line in text_paths:
            file.write(f"{line}\n")

    columns = ["audio_id", "audio", "raw_text", "normalized_text", "gender", "speaker_id"]

    df = pd.DataFrame(columns=columns)
    audio_id = []
    path = []
    with open("/home/zhuldyz/Downloads/KazSpeechT5/audio_paths.txt", 'r') as file:
        for line in file:
            audio_id.append(line[50:-5])
            path.append(line[:-1])

    df["audio_id"] = audio_id

    speaker_id = []
    gender = []
    with open("/home/zhuldyz/Downloads/KazSpeechT5/speakers_id.txt", 'r') as file:
        for line in file:
            speaker_id.append(line[:-1])
            if line != 'M2' or line !='M1':
                gender.append('female')
            else:
                gender.append('male')

    df["speaker_id"] = speaker_id
    df["gender"] = gender

    raw_texts = []
    with open('/home/zhuldyz/Downloads/KazSpeechT5/transcripts_paths.txt', 'r') as file:
        for line in file:
            with open(line[:-1], 'r') as file:
                raw_texts.append(file.read())

    df["raw_text"] = raw_texts

    df["normalized_text"] = df["raw_text"].str.lower()

    # with open("/home/zhuldyz/Downloads/KazSpeechT5/audio_paths.txt", 'r') as file:
    audio = []
    with open("/home/zhuldyz/Downloads/KazSpeechT5/audio_paths.txt", 'r') as file:
        for line in file:
            audio.append(line[:-1])

    df['audio'] = audio


    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

    def extract_all_chars(batch):
        all_text = " ".join(batch["normalized_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )

    dataset_vocab = set(vocabs["vocab"][0])

    checkpoint = "microsoft/speecht5_tts"
    processor = SpeechT5Processor.from_pretrained(checkpoint)
    tokenizer = processor.tokenizer
    tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}

    # print(dataset_vocab - tokenizer_vocab)

    # code.interact(local=dict(globals(), **locals()))

    return dataset_vocab, tokenizer_vocab, dataset


if __name__ == '__main__':
    _ = create_dataset()