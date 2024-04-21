from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaModel, RobertaTokenizer
import torch
import numpy as np

checkpoint = "facebook/nllb-200-distilled-600M"

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

target_lang = "eng_Latn"; source_lang = 'arb_Arab'

translator_en = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang, max_length = 50)
translator_ar = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=target_lang, tgt_lang=source_lang, max_length = 50)



def translate_to_eng(sent):
  return translator_en(sent)[0]['translation_text']

def translate_to_ara(sent):
  return translator_ar(sent)[0]['translation_text']

model = RobertaModel.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def calculate_similarity(text1,text2):
    inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
      output1 = model(**inputs1)
      output2 = model(**inputs2)

    embeddings1 = output1.last_hidden_state.mean(dim=1)
    embeddings2 = output2.last_hidden_state.mean(dim=1)

    similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1, axis=1)[:, None] * np.linalg.norm(embeddings2, axis=1))
    return similarity[0][0]
