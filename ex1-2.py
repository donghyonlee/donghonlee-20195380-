import pandas as pd
import datetime
import numpy as np

data = pd.read_csv("C:/Users/admin/Desktop/hospital_billing.csv")
df = pd.DataFrame(data)

df_r1 = df.rename(columns={'case_id':'caseid','timestamp': 'ts'}, inplace= True)
df['activity'] = df['activity'].astype("category")
df['ts'] =df['ts'].astype('datetime64[ns]')

df_r2=df.loc[:,['caseid','activity','ts']]

df_ex4 = df_r2.groupby('caseid', sort = 'ts')['activity'].apply(list)

##5 start
##5 start
duration = []
caseidlist= []
df_interval = df_r2.groupby('caseid',sort = 'ts')['ts'].apply(list)
for i in df_interval:
    s1 = datetime.datetime.strptime(i[0],'%Y/%m/%d %H:%M:%S.%f')
    s2 = datetime.datetime.strptime(i[-1],'%Y/%m/%d %H:%M:%S.%f')
    t_diff = (s2-s1).total_seconds()/60
    duration.append(t_diff)

event_number = []
for i in df_ex4:
    event_number.append(len(i))

df_new = pd.DataFrame(columns=['caseid','duration','event_number'])

df_new['caseid'] = df_interval.index.values
df_new['duration'] =duration
df_new['event_number'] = event_number
##print(df_new)

maxduration = max(list(df_new['duration']))
minduration = min(list(df_new['duration']))
whatismax = []

for pos,m in enumerate(list(df_new['duration'])):
    if m == maxduration:
        whatismax.append(list(df_new['caseid'])[pos])

whatismin = []

for pos,m in enumerate(list(df_new['duration'])):
    if m == minduration:
        whatismin.append(list(df_new['caseid'])[pos])


##7 start
whatismaxnum = []
maxnumber = max(list(df_new['event_number']))
for pos,m in enumerate(list(df_new['event_number'])):
    if m == maxnumber:
        whatismaxnum.append(list(df_new['caseid'])[pos])

