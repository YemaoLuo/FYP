import pandas as pd
from tqdm import tqdm

from api.characteristic_api import get_mask_sentence
from api.completion_api import get_completion

with open('./test.txt', 'r', encoding='utf-8') as f:
    text = f.read()

sentences = [s.strip() + '。' for s in text.split('。') if s.strip()]

print('masking...')
mask_sentences = [get_mask_sentence(s) for s in tqdm(sentences)]
write_data = []
print('completing...')
for s in tqdm(mask_sentences):
    completions = get_completion(s, 5, 'gpt')
    write_data.append(completions)
    write_data.append([''])

write_data_flat = [item for sublist in write_data for item in sublist]

output_file = "./raw_data.xlsx"
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df = pd.DataFrame(write_data_flat)
    df.to_excel(writer, index=False, header=False)
