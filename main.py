import whisper
import data
import time

samples = data.get()
model = whisper.load_model("base")

for sample in samples:
  start_time = time.time()
  result = model.transcribe(sample['fileName'])
  sample['transcribedText'] = result["text"]
  sample['runtime'] = (time.time() - start_time)
  print(sample)
