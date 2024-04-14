from transformers import SpeechT5ForTextToSpeech
from transformers import SpeechT5Processor


checkpoint = "microsoft/speecht5_tts"
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
processor = SpeechT5Processor.from_pretrained(checkpoint)
