import csv

from tqdm import tqdm

from api.translate_api import *

with open('./test.txt', 'r', encoding='utf-8') as f:
    text = f.read()

sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
for s in sentences:
    print(s)

print('=' * 100)
print('translating...')

data = []
for i in tqdm(range(len(sentences))):
    baidu_translation = baidu_translate(sentences[i], 'en', 'zh')
    google_translation = google_translate(baidu_translation, 'zh-CN', 'en')
    line = [sentences[i], baidu_translation, google_translation]
    data.append(line)

print('=' * 100)
print('writing...')

filename = './preprocessing_data.csv'

with open(filename, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)
