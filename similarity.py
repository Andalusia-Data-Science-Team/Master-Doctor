from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import pandas as pd
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(sentences):
    tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model(**tokenized)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

def dummy_func(sent):
  return sent

def find_bert_similarity(main_names=list,tar_names = list, tar_keys = [], preprocess_func=dummy_func,columns_names=[],sorted=True,sim_thresh=0,csv_name='bert_matches'):
    main_engine = []; possible_match = []; keys_match = []; scores_sim= []; best_matches_df = pd.DataFrame()

    main_names_prep = [dummy_func(name) for name in main_names]
    embeddings1 = get_bert_embeddings(tar_names)
    embeddings2 = get_bert_embeddings(main_names_prep)

    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    most_similar_indices = np.argmax(similarity_matrix, axis=1)

    index_mapping = {i: most_similar_indices[i] for i in range(len(tar_names))}

    for i, item in enumerate(tar_names):
      most_similar_item = main_names[index_mapping[i]]
      similarity_score = similarity_matrix[i][index_mapping[i]]
      main_engine.append(most_similar_item); possible_match.append(item); scores_sim.append(similarity_score)

    if tar_keys != []:
      for item in tar_names:
        index = tar_names.index(item)
        key = tar_keys[index]
        keys_match.append(key)

    if columns_names != []:
      columns = columns_names
    else:
      columns = ['Main_Engine', 'Secondary_Search', 'Secondary_Key', 'Similarity_Score']
    columns[-1] = "Similarity_Score"
    best_matches_df[columns[0]] = main_engine; best_matches_df[columns[1]] = possible_match
    
    if tar_keys != []:
      best_matches_df[columns[2]] = keys_match
      best_matches_df[columns[3]] = scores_sim
    else:
      best_matches_df[columns[2]] = scores_sim

    if sorted:
      best_matches_df.sort_values(by=['Similarity_Score'], ascending=False, inplace=True)

    best_matches_df = best_matches_df[best_matches_df['Similarity_Score'] > sim_thresh]
    best_matches_df = best_matches_df.drop_duplicates()

    csv_name = csv_name+ '.csv'
    best_matches_df.to_csv(csv_name,index=False,encoding='utf-8-sig')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_sim(text1,text2):
    combined_text = text1 + text2
    vectorizer = TfidfVectorizer()

    combined_text = [str(element) for element in combined_text]
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    num_text1 = len(text1)
    similarity_scores = cosine_similarities[:num_text1, num_text1:]

    return similarity_scores

def dummy_func(sent):
  return sent

def find_cosine_similarity(main_names=list,tar_names=list,preprocess_func = dummy_func,tar_keys=[],columns_names=[],exact_matching=True,csv_name ='best_matches',sorted=True,sim_thresh=0):
  
  best_matches_df = pd.DataFrame(); main_engine = []; possible_match = []; keys_match = []; scores_sim= []

  data_limit = len(main_names)
  looper = data_limit // 1000 + 1
  main_prep = [preprocess_func(sent) for sent in main_names]

  for i in range(looper):
    start_index = i*1000; end_index = min(start_index+1000,data_limit)

    if end_index == data_limit:
        text1_patch = main_prep
    else:
        text1_patch = main_prep[start_index:end_index]
    text2_patch = tar_names[:]

    if text1_patch != []:
        similarity_scores = calculate_sim(text1_patch,text2_patch)

        for i, scores in enumerate(similarity_scores):
            if len(scores) > 0:
                best_match_index = max(range(len(scores)), key=scores.__getitem__)

                best_match_score = scores[best_match_index]
                best_match_text = tar_names[best_match_index]
                if tar_keys != []:
                  best_match_key = tar_keys[best_match_index]
                  keys_match.append(best_match_key)

                main_engine.append(main_names[i]);  possible_match.append(best_match_text);  scores_sim.append(best_match_score)

  ## check for exact matches
  if exact_matching:
    for i in range(len(main_engine)):
      main_elem = main_engine[i].lower()
      for j in range(len(tar_names)):
        tar_elem = tar_names[j].strip().lower()
        if main_elem == tar_elem:
          possible_match[i] = tar_names[j]
          scores_sim[i] = 1.0
          if tar_keys != []:
             keys_match[i] = tar_keys[j]


  if columns_names != []:
    columns = columns_names
  else:
    columns = ['Main_Engine', 'Secondary_Search', 'Secondary_Key', 'Similarity_Score']
  columns[-1] = "Similarity_Score"

  best_matches_df[columns[0]] = main_engine; best_matches_df[columns[1]] = possible_match

  if tar_keys != []:
    best_matches_df[columns[2]] = keys_match
    best_matches_df[columns[3]] = scores_sim
  else:
    best_matches_df[columns[2]] = scores_sim

  if sorted:
    best_matches_df.sort_values(by=['Similarity_Score'], ascending=False, inplace=True)

  best_matches_df = best_matches_df[best_matches_df['Similarity_Score'] > sim_thresh]
  best_matches_df = best_matches_df.drop_duplicates()

  csv_name = csv_name+ '.csv'
  best_matches_df.to_csv(csv_name,index=False,encoding='utf-8-sig')


def find_cosine_similarity_df(df_main,df_tar,main_col_name=str,tar_col_name=str,key_col_name=[]):
  main_names = df_main[main_col_name].tolist();  tar_names = df_tar[tar_col_name].tolist()
  if key_col_name !=[]:
      tar_keys = df_tar[key_col_name]
      find_cosine_similarity(main_names=main_names,tar_names=tar_names,preprocess_func = dummy_func,tar_keys=tar_keys,columns_names=[],exact_matching=True,csv_name ='best_matches',sorted=True,sim_thresh=0)
  else:
      find_cosine_similarity(main_names=main_names,tar_names=tar_names,preprocess_func = dummy_func,tar_keys=[],columns_names=[],exact_matching=True,csv_name ='best_matches',sorted=True,sim_thresh=0)


from Levenshtein import distance as lev

def calculate_levenshtein(text1,text2):
  return lev(text1,text2)

def dummy_func(sent):
  return sent

def find_lev_difference(main_names = list,tar_names = list, tar_keys = [], preprocess_func=dummy_func,columns_names=[],sorted=True,diff_thresh=100,csv_name='lev_difference'):
  
  main_engine = []; possible_match = []; keys_match = []; scores_sim= []; best_matches_df = pd.DataFrame()
  for i in range(len(main_names)):

    min_val = float(99999); main_elem = main_names[i]; main_engine.append(main_elem)
    main_elem = preprocess_func(main_elem)

    for j in range(len(tar_names)):
        tar_match = tar_names[j]
        difference = calculate_levenshtein(main_elem,tar_match)

        if (difference < min_val):
          if tar_keys == []:
            key = 0
          else:
            key = tar_keys[j]

          row = [tar_match,key,difference]
          min_val = difference

    possible_match.append(row[0]); keys_match.append(row[1]); scores_sim.append(row[2])

  if columns_names != []:
    columns = columns_names
  else:
    columns = ['Main_Engine', 'Secondary_Search', 'Secondary_Key', 'Difference_Score']
  columns[-1] = "Difference_Score"

  best_matches_df[columns[0]] = main_engine; best_matches_df[columns[1]] = possible_match
  if tar_keys[0] != 0:
    best_matches_df[columns[2]] = keys_match
    best_matches_df[columns[3]] = scores_sim
  else:
    best_matches_df[columns[2]] = scores_sim

  if sorted:
    best_matches_df.sort_values(by=['Difference_Score'], ascending=True, inplace=True)

  best_matches_df = best_matches_df[best_matches_df['Difference_Score'] < diff_thresh]
  best_matches_df = best_matches_df.drop_duplicates()

  csv_name = csv_name+ '.csv'
  best_matches_df.to_csv(csv_name,index=False,encoding='utf-8-sig')
