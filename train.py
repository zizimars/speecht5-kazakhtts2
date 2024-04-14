from collator import data_collator
from model import model, processor
from functools import partial
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, Trainer
from speaker_speech import dataset_creator
import huggingface_hub

huggingface_hub.login('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')

dataset = dataset_creator()
dataset = dataset.train_test_split(test_size=0.1)

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(model.generate, use_cache=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_kazakh_tts2",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=1000,
    max_steps=5000,
    gradient_checkpointing=False,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    # report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

trainer.train()

trainer.push_to_hub()