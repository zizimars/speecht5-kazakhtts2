import os
import torch
# from speechbrain.pretrained import EncoderClassifier
from speechbrain.inference.speaker import EncoderClassifier
from transformers import SpeechT5Processor
from transliteration import transliterate_text
from model import processor

import code

def dataset_creator():
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    dataset = transliterate_text()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", spk_model_name),
    )


    def create_speaker_embedding(waveform):
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings

    def prepare_dataset(example):
        audio = example["audio"]

        example = processor(
            text=example["normalized_text"],
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )

        # strip off the batch dimension
        example["labels"] = example["labels"][0]

        # use SpeechBrain to obtain x-vector
        example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

        return example

    # processed_example = prepare_dataset(dataset[0])
    # code.interact(local=dict(globals(), **locals()))
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    
    def is_not_too_long(input_ids):
        input_length = len(input_ids)
        return input_length < 200

    dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
    return dataset