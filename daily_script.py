import urllib
import sqlalchemy
import pyodbc
import os
from datetime import datetime, timedelta
from datetime import date
import pandas as pd
import calendar
from similarity import calculate_levenshtein
from utilities import calculate_sim,preprocess_doc,find_kth_max_indices,fuzzy_score,data_dict_utools
from utilities import check_ara, translate_ara
import time

seconds_in_a_day = 24 * 60 * 60  # Total seconds in a day
sleep_duration_days = 0.75
sleep_duration_seconds = sleep_duration_days * seconds_in_a_day

query = f'''
select distinct o.OrganizationUnifiedName,s.Staff_name from dax.factbilling b inner join dax.dimstaff s
on b.Doctor_Key=s.Staff_Key
inner join dax.dimOrganization o on b.Organization_Key=o.OrganizationKey
where
service_date>cast(DATEADD(month, DATEDIFF(month, -1, getdate()) - 2, 0)as date)
and  b.organization_key in (1,3,5,4,9,10,13)
order by o.OrganizationUnifiedName
'''

def update_table(df):
    connect_string = urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};Server=AWS-BI-03;Database=DWH;UID=AI;PWD=P@ssw0rd;charset="utf-8";')
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    with engine.connect() as connection:
        df.to_sql('Aff_doc_mapping', connection, index=False,if_exists='append',chunksize=100,schema='dbo')

def load_lead_src_data():
    connect_string = urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};Server=AWS-BI-03;Database=DWH;UID=AI;PWD=P@ssw0rd;charset="utf-8";')
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    query = f"""
select distinct Source_Unit, Topic from dimlead
where Source_Unit in ('HJH','ALW','ADC','APC','AKW')
and Lead_Source in ('DoctorUna','EasyDoc','Tebcan','Vezeeta')
and Lead_CreationDate>=  cast(DATEADD(mm, DATEDIFF(m,0,GETDATE()),0)as date)
and topic not in (select [Doctor Topic]from Aff_doc_mapping)
order by Source_Unit
"""

    with engine.connect() as connection:
        return pd.read_sql(query, engine)

def load_staff_data():
    connect_string = urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};Server=AWS-BI-03;Database=DWH;UID=AI;PWD=P@ssw0rd;charset="utf-8";')
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    with engine.connect() as connection:
        return pd.read_sql(query, engine)

df_Staff = load_staff_data()

df_Staff.dropna(inplace=True)
today = date.today();  month_num = str(today.month); year_num = str(today.year)
path_staff = f'data/Billing/{year_num}_{month_num}'

if not os.path.exists(path_staff):
    os.makedirs(path_staff)

path_staff = path_staff+'/staff_data.csv'
df_Staff.to_csv(f'data/Billing/{year_num}_{month_num}/staff_data.csv',
                index=False,encoding='utf-8-sig')  ## staff add translation



df_lead = load_lead_src_data()

df_lead.Source_Unit = df_lead.Source_Unit.str.replace('APC','AFW')
df_lead = df_lead[df_lead['Topic']!='General']

today = date.today(); day_num =str(today.day);  month_num = str(today.month); year_num = str(today.year)
dated_dir = year_num + '_' + month_num + '/' + day_num
path_dir = 'data/CRM/' + dated_dir

if not os.path.exists(path_dir):
    os.makedirs(path_dir)

path_dir = path_dir + '/lead_source_doc.csv'
df_lead.to_csv(path_dir,index=False,encoding='utf-8-sig') ## lead_src add translation

today = date.today();   month_num = str(today.month); year_num = str(today.year)
if len(month_num) < 2:
    month_num = '0' +str(month_num)

month_name = calendar.month_name[int(month_num)]
data_dict = data_dict_utools

lead_src_path = 'data/CRM/' + year_num + '_' + month_num + '/'
path_crm = lead_src_path + 'lead_source_doc.csv'

doc_undefined_prep = pd.read_csv(path_dir)
doc_undefined_unprep = pd.read_csv(path_dir)

origin_staff  = pd.read_csv(path_staff)
edited_staff  = pd.read_csv(path_staff)

for i in range(len(doc_undefined_prep)):
    if check_ara(doc_undefined_prep.Topic.iloc[i]):
        doc_undefined_prep.Topic.iloc[i] = translate_ara(doc_undefined_prep.Topic.iloc[i])

## main preprocessing
edited_staff['Staff_name'] = edited_staff['Staff_name'].apply(lambda x: preprocess_doc(x))
doc_undefined_prep['Topic'] = doc_undefined_prep['Topic'].apply(lambda x: preprocess_doc(x))

num_patches = 1; patch_size = len(doc_undefined_prep) // num_patches
all_bu_set = set(doc_undefined_prep['Source_Unit'])

best_matches_df = pd.DataFrame(); doctors = []; doctors_prep=[]; poss_match = []; scores_sim= []; bu_match = []

for bu in all_bu_set:
    bu_dict = data_dict[bu]

    df_lbu = doc_undefined_prep[doc_undefined_prep['Source_Unit']==bu]
    df_bu  = doc_undefined_unprep[doc_undefined_unprep['Source_Unit']==bu]
    df_sbu = edited_staff[edited_staff['OrganizationUnifiedName']==bu]
    df_org_sbu = origin_staff[origin_staff['OrganizationUnifiedName']==bu]


    keys_done = []
    for counter_ in range(len(df_lbu.Topic)):
        name = df_lbu.Topic.iloc[counter_]
        if name in bu_dict.keys():
            keys_done.append(df_bu.Topic.iloc[counter_])
            best_match_text = bu_dict[name]
            doctors.append(df_bu.Topic.iloc[counter_]); poss_match.append(best_match_text)
            extract_score = fuzzy_score()
            doctors_prep.append(df_lbu.Topic.iloc[counter_]); scores_sim.append(extract_score); bu_match.append(bu)

    for patch_index in range(num_patches):
        start_index = patch_index * patch_size
        end_index = (patch_index + 1) * patch_size if patch_index < num_patches - 1 else len(df_bu)
        text1_patch = df_lbu['Topic'].tolist()[start_index:end_index]
        text2_patch = df_sbu['Staff_name'].tolist()

        similarity_scores = calculate_sim(text1_patch,text2_patch)
        for i, scores in enumerate(similarity_scores):
            if len(scores)>0:
                    indices_matches = find_kth_max_indices(scores, 10)
                    best_match_index = max(range(len(scores)), key=scores.__getitem__)

                    match_indices = list(reversed(list(indices_matches)))
                    best_match_score = scores[best_match_index]
                    best_match_text = df_org_sbu['Staff_name'].iloc[best_match_index]

                    if len(df_lbu.Topic.iloc[i].split(' '))>1:
                        oth_cond = df_lbu.Topic.iloc[i].split(' ')[1].lower().strip() in best_match_text.lower().split(' ')[1:]
                    else:
                        oth_cond = df_lbu.Topic.iloc[i].split(' ')[0].lower().strip() in best_match_text.lower().split(' ')[1:]

                    text1 = df_lbu.Topic.iloc[i].split(' ')[0].lower(); text2 = best_match_text.split(' ')[0].lower()

                    if calculate_levenshtein(text1,text2) < 3 or (text1==text2 and oth_cond):
                        doctors.append(df_bu.Topic.iloc[i]); poss_match.append(best_match_text)
                        doctors_prep.append(df_lbu.Topic.iloc[i]); scores_sim.append(best_match_score); bu_match.append(bu)

                    else:
                        topic_text = df_bu.Topic.iloc[i]; best_match_ex = best_match_text
                        prep_text  = df_lbu.Topic.iloc[i]
                        flag = 0; counter =0

                        text1 = df_bu['Topic'].iloc[i].split(' ')[0].lower()

                        while counter<5 and flag==0:
                            for looper in range(len(df_sbu)):
                                text2 = df_sbu['Staff_name'].iloc[looper].split(' ')[0].lower()
                                if calculate_levenshtein(text1,text2) == counter or text1==text2:
                                    flag = 1
                                    best_match_ex  = df_org_sbu['Staff_name'].iloc[looper]
                            counter += 1
                        doctors.append(topic_text); poss_match.append(best_match_ex)
                        doctors_prep.append(prep_text)
                        scores_sim.append(best_match_score); bu_match.append(bu)

columns = ['Doctor Topic', 'Doctor Prep', 'Staff Name','Business Unit', 'Similarity Score']
best_matches_df[columns[0]] = doctors #; best_matches_df[columns[1]] = doctors_prep
best_matches_df[columns[2]] = poss_match
best_matches_df[columns[3]] = bu_match;  best_matches_df[columns[4]] = scores_sim

best_matches_df.sort_values(by=['Similarity Score'],ascending=False,inplace=True)
best_matches_df.drop_duplicates(subset=['Doctor Topic'],keep='first',inplace=True)

matches_path = 'output/' + year_num + '_' + month_num + '_' + day_num + '_' + 'best_matches.csv'
best_matches_df.to_csv(matches_path,index=False,encoding='utf-8-sig')
best_matches_df['Staff Key'] = 'None'

#print(best_matches_df)
update_table(best_matches_df)
today_sch = 'Process done for scheduled mapping on ' + day_num + '/'  + month_num + '/' + year_num + '.'
print(today_sch)