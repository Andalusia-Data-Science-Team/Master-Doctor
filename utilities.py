from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import openpyxl
import urllib
import sqlalchemy
import pyodbc
import numpy as np
import json
import re
from googletrans import Translator

with open('data/dict_names.json', 'r') as json_file:
    data_dict_utools = json.load(json_file)

specialties = pd.read_excel('data/unified_speciality_names.xlsx')['Unified Specialty'].tolist()
ara_char = 'د ج ح خ ه ع غ ف ق ث ص ض ش س ي ب ل ا ت ن م ك ط ز ظ و ة ى ل ا ر ؤ ء ئ'.split(' ')

translator = Translator()


def translate_ara(text):
    target_language = 'en'
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text

def fuzzy_score():
    return pd.Series(np.random.uniform(0.90, 1.0))[0]

def find_kth_max_indices(arr, k):
    if len(arr) < k:
        raise ValueError("Array does not have enough elements for the specified k.")

    max_indices = (-arr).argsort()[:k]
    return max_indices

def load_lead_src_data(year,month,day,day2):
    connect_string = urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};Server=AWS-BI-01;Database=DWH;UID=sa;PWD=BI#AHQ;charset="utf-8";')
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    query = f"""
SELECT TOPIC,Lead_Source,Source_Unit
FROM DimLead
WHERE Lead_Source IN ('VEZEETA','TEBCAN','EASYDOC','Doctoruna')
AND YEAR(Lead_CreationDate) = {year}
AND MONTH(Lead_CreationDate) = {month}
AND DAY(Lead_CreationDate) = {day}
and Source_Unit in ('HJH','AKW','ALW','ADC','APC')
"""

    with engine.connect() as connection:
        return pd.read_sql(query, engine)


def preprocess_doc(doc):

    #doc = str(doc).split('-')[0] ## string assertion
    for specialty in specialties:   ## eliminate specialties from doc names
       if specialty in doc:
           doc.replace(specialty,'')

    doc = doc.split('||')[0]    ## || is used to split name, from date, from specialty.
    doc = doc.split('2')[0]     ## Dates start usually with 2023 and are put after names.
    doc = doc.replace('-','')
    doc = doc.replace("'s",'')  ## replace 's with None.

    spaces = [' '*i for i in range(1,6)][-1::-1][:-1]
    for sp in spaces:
        doc = doc.replace(sp,' ')
    doc = doc.strip()


    dict_names = {'Amin':"Ameen", 'Al ':'El', 'EL ':"El",'AL ':'El'," Al":" El",'El ':'El','Abdul ':'Abdel',"Al -":"Al",
                  'Abdel ':'Abdel',"ABDEL ":'Abdel','Mojab': 'Moajeb', 'Abd ':"Abd",'Abu ':'Abu','Abou ':'Abu',
                  'Seif eldin':"SAIFELDEEN",'Seif eldeen':"SAIFELDEEN", 'Seif el deen':"SAIFELDEEN","Jamil":"Gameel",
                  "Danyh":"Dania"
                  }

    for elem in dict_names.keys():
        if elem in str(doc):
            doc = doc.replace(elem,dict_names[elem])
    return doc



def check_ara(prep_doc):
    for ara in ara_char:
        if ara in prep_doc:
            return 1
    return 0


def calculate_sim(text1,text2):
    # Combine text data from both patches
    combined_text = text1 + text2

    vectorizer = TfidfVectorizer()

    combined_text = [str(element) for element in combined_text]

    tfidf_matrix = vectorizer.fit_transform(combined_text)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    num_text1 = len(text1)
    similarity_scores = cosine_similarities[:num_text1, num_text1:]

    return similarity_scores

def normalize_arab(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return text

def normalize_name(name):
    name = re.sub(r' AL ', ' EL', name)
    name = re.sub(r'^AL ', 'EL ', name)
    name = re.sub(r'^EL ', 'EL', name)
    name = re.sub(r' EL ', ' EL', name)
    name = re.sub(r'^ABU ', 'ABU', name)
    name = re.sub(r' ABU ', ' ABU', name)
    name = re.sub(r'^ABO ', 'ABO', name)
    name = re.sub(r' ABO ', ' ABO', name)
    name = re.sub(r' BN ', ' BN', name)
    name = re.sub(r' ABD ', ' ABD', name)
    name = re.sub(r'^ABD ', 'ABD', name)
    name = re.sub(r'^ABDEL ', 'ABDEL', name)
    name = re.sub(r' ABDEL ', ' ABDEL', name)
    name = re.sub(r' ALLAH ', 'ALLAH ', name)
    name = re.sub(r' ALLAH$', 'ALLAH', name)

    name = re.sub(r'ELDE.N', 'ELDEN', name)
    name = re.sub(r'KLTHOM', 'KALSOM', name)
    name = re.sub(r'^OM ', 'OM', name)
    name = re.sub(r'ESSRAA', 'ESRAA', name)
    name = re.sub(r'ESRAA', 'ESRAA', name)
    name = re.sub(r'HAMID', 'HAMED', name)
    name = re.sub(r'HAMEED', 'HAMED', name)
    name = re.sub(r'JODY', 'GODY', name)
    name = re.sub(r'HODA', 'HUDA', name)
    name = re.sub(r'MOHAMED', 'MOHAMMED', name)
    name = re.sub(r'MOHAMD', 'MOHAMMED', name)
    name = re.sub(r'MOHMOUD', 'MAHMOUD', name)
    name = re.sub(r'MOHM.D', 'MAHMOUD', name)
    name = re.sub(r'YASIEN', 'YASSEN', name)
    name = re.sub(r'YAS.N', 'YASSEN', name)
    name = re.sub(r'ZENHOM', 'ZANHOM', name)
    name = re.sub(r'SAMEER', 'SAMIR', name)
    name = re.sub(r'YASER', 'YASSER', name)
    name = re.sub(r'ISMAEL', 'ISMAIL', name)
    name = re.sub(r'FAWZ.A', 'FAWZEYA', name)
    name = re.sub(r'WALAY', 'WALY', name)
    name = re.sub(r'YOUSEF', 'YOUSSEF', name)
    name = re.sub(r'YOS.F', 'YOUSSEF', name)
    name = re.sub(r'SAID', 'SAYED', name)
    name = re.sub(r'SAIED', 'SAYED', name)
    name = re.sub(r'MOUSTAFA', 'MOSTAFA', name)
    name = re.sub(r'MUSTAFA', 'MOSTAFA', name)
    name = re.sub(r'KAREM', 'KARIM', name)
    name = re.sub(r'KAREEM', 'KARIM', name)
    name = re.sub(r'ZEINAB', 'ZINAB', name)
    name = re.sub(r'ZENAB', 'ZINAB', name)
    name = re.sub(r'NAREMAN', 'NARIMAN', name)
    name = re.sub(r'NAD.A', 'NADIA', name)
    name = re.sub(r'SOUZAN', 'SUZAN', name)
    name = re.sub(r' SH.REEF ', ' SHERIF ', name)
    name = re.sub(r' SH.REF ', ' SHERIF ', name)
    name = re.sub(r' SHEREF ', ' SHERIF ', name)
    name = re.sub(r' SHREEF ', ' SHERIF ', name)
    name = re.sub(r' SHR.F ', ' SHERIF ', name)
    name = re.sub(r' SHREF ', ' SHERIF ', name)

    return name.strip()