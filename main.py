import whisper
import data
import time
import json

samples = data.get()
model = whisper.load_model("base")

def transcribe (sample, model):
  start_time = time.time()
  result = model.transcribe(sample['file-name'])
  sample['runtime'] = (time.time() - start_time)
  sample['transcribed-text'] = result["text"]

def calculate_text_lens (sample):
  sample['transcribed-text-size'] = len(sample['transcribed-text'].split())
  sample['text-size'] = len(sample['text'].split())

def find_wrong_words (sample):
  transTextList = sample['transcribed-text'].split()
  textList = sample['text'].split()
  sample['wrong-words'] = [x for x in transTextList + textList if x not in transTextList or x not in textList]

def calculate_wrong_words_percent (sample):
  sample['error-percentage'] = (sample['text-size'] / 100) * len(sample['wrong-words'])

for sample in samples:
  transcribe(sample, model)
  calculate_text_lens(sample)
  find_wrong_words(sample)
  calculate_wrong_words_percent(sample)

print(json.dumps(samples, indent=2, sort_keys=False))