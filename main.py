import whisper
import data
import time
import csv
import matplotlib.pyplot as plt

model_types = ['base', 'small', 'medium']

def bar_chart(data):
  pos = list(range(10))
  plt.bar(pos, data['errors'], color='blue')
  plt.xticks(ticks=pos, labels=data['regions'])
  plt.title("Tipo de modelo: " + data['model_type'])
  plt.xlabel('Região')
  plt.ylabel('Porcentagem de erro das amostras')
  plt.show()

def transcribe (sample, model):
  start_time = time.time()
  result = model.transcribe(sample['file-name'])
  sample['runtime'] = round((time.time() - start_time), 2)
  sample['transcribed-text'] = result["text"]

def calculate_text_lens (sample):
  sample['transcribed-text-size'] = len(sample['transcribed-text'].split())
  sample['text-size'] = len(sample['text'].split())

def find_wrong_words (sample):
  transTextList = sample['transcribed-text'].split()
  textList = sample['text'].split()
  sample['wrong-words'] = [x for x in transTextList if x not in textList]

def calculate_wrong_words_percent (sample):
  sample['number-wrong-words'] = len(sample['wrong-words'])
  sample['error-percentage'] = round((sample['number-wrong-words'] * 100) / (sample['text-size']), 2)


array = []

for model_type in model_types:
  samples = data.get()
  model = whisper.load_model(model_type)

  regions = []
  errors = []

  for sample in samples:
    transcribe(sample, model)
    calculate_text_lens(sample)
    find_wrong_words(sample)
    calculate_wrong_words_percent(sample)

    regions.append(sample['region'])
    errors.append(sample['error-percentage'])

  array.append({
    'model_type': model_type,
    'regions': regions,
    'errors': errors
  })

  fields = [
    'region',
    'text',
    'file-name',
    'transcribed-text',
    'runtime',
    'text-size',
    'transcribed-text-size',
    'wrong-words',
    'number-wrong-words',
    'error-percentage',
    'lang'
  ]

  filename = model_type + '.csv'

  with open (filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writeheader()
    writer.writerows(samples)

for dt in array:
  bar_chart(dt)