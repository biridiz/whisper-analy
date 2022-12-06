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


# Deve calcular o tamanho (quantidade de palavras)
# e comparar com o tamanho da amostra

# Deve calcular quantidade de erros para cada amostra

# Deve calcular porcentagem de erros para cada amostra
