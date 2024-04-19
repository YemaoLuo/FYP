import csv

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

from api.characteristic_api import get_mask_sentence
from api.completion_api import get_completion


def get_gpt_completion(mask_sentence, count):
    return get_completion(mask_sentence, count, 'gpt')


def get_bert_completion(mask_sentence, count):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')

    input_ids = tokenizer.encode(mask_sentence, add_special_tokens=True, return_tensors='pt')
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

    mask_predictions = predictions[0, mask_token_index, :]
    top_predictions = torch.topk(mask_predictions, k=count, dim=1).indices[0].tolist()

    completed_sentences = []
    for prediction in top_predictions:
        predicted_token = tokenizer.convert_ids_to_tokens([prediction])[0]
        completed_sentence = mask_sentence.replace(tokenizer.mask_token, predicted_token)
        completed_sentences.append(completed_sentence)

    return completed_sentences


def read_sentences_from_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            sentences.append(sentence)
    return sentences


if __name__ == '__main__':
    sentences = read_sentences_from_file("sample.txt")
    mask_sentences = []
    data = []
    for sentence in tqdm(sentences):
        mask_sentences.append(get_mask_sentence(sentence))
    for mask_sentence in tqdm(mask_sentences):
        data.append([mask_sentence])
        data.append(["LLM"])
        for gpt in get_gpt_completion(mask_sentence, 5):
            data.append([gpt])
        data.append(["BERT"])
        for bert in get_bert_completion(mask_sentence, 5):
            data.append([bert])
        data.append(["*****************************************"])

    filename = 'completion_data.csv'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
