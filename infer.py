import torch
from collator import data_collator
from model import model, processor
from functools import partial
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, Trainer
from speaker_speech import dataset_creator
import huggingface_hub
from transformers import SpeechT5ForTextToSpeech
from transformers import SpeechT5Processor


dataset = dataset_creator()
dataset = dataset.train_test_split(test_size=0.1)

example = dataset["test"][0]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)

text = "kel balalar okylyk"
inputs = processor(text=text, return_tensors="pt")


model = SpeechT5ForTextToSpeech.from_pretrained(
    "/home/zhuldyz/Downloads/KazSpeechT5/speecht5_finetuned_voxpopuli_nl/checkpoint-10",
    local_files_only=True,
)

from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)


from IPython.display import Audio

Audio(speech.numpy(), rate=16000)