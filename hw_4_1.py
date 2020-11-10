#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

event_log = pd.read_csv('C:/Users/admin/Desktop/Large_hw4.csv')


# In[5]:


columns = ['caseid', 'activity', 'ts', 'resource', 'outcome']
df = event_log[['Case', 'Activity', 'Timestamp', 'Resource', 'is_anomalous']]
df = df.rename(columns={v:columns[i] for i, v in enumerate(df.columns)})

df.shape


# In[6]:


df


# In[7]:


from sklearn import preprocessing
encoder_a = preprocessing.LabelEncoder()
encoder_r = preprocessing.LabelEncoder()
activity_ids = encoder_a.fit_transform(df.activity)
resources = encoder_r.fit_transform(df.resource)
df.activity = activity_ids
df.resource = resources
df.ts = pd.to_datetime(df.ts)

df


# In[8]:


def process_data(data, L):

    data['one'] = 1
    data['order'] = data.groupby('caseid')['one'].transform(pd.Series.cumsum)
    data= data.loc[data['order'] <= L].reset_index(drop=True)
    data['max'] = data.groupby('caseid')['order'].transform(max)
    data = data.loc[data['max'] == L].reset_index(drop=True)
    data = data.drop(columns=['one', 'order', 'max'])



    data['duration'] = data.groupby('caseid')['ts'].transform(lambda x: x.diff())
    data['duration'] = data['duration'].apply(lambda x: x.total_seconds())
    data['duration'] = data['duration'].fillna(0)

    data['aggregation'] = data[['activity', 'duration', 'resource']].apply(lambda x: list(x), axis=1)
    params = {
        'aggregation': lambda x: sum(x, [])
    }
    new_atts = data.groupby('caseid').agg(params).reset_index()
    new_atts = pd.merge(new_atts, data[['caseid','outcome']].drop_duplicates(), on='caseid', how="left")
    column_name = []
    for i in range(0,L):
        column_name.append('act_{}'.format(i))
        column_name.append('duration_{}'.format(i))
        column_name.append('res_{}'.format(i))
    new_atts[column_name] = pd.DataFrame(new_atts.aggregation.tolist(), index=new_atts.index)
    new_atts = new_atts.drop(columns=['aggregation'])

    result = new_atts

    return result


# In[9]:


from collections import Counter

def augment_data(data):
  columns=['n_work_items', 'n_curr', 'per_case', 'per_curr_ho']
  result = []

  resources = data[[i for i in data.columns if "res" in i or "outcome" in i]]

  for i in resources.index:
    row = resources.iloc[i]
    df = resources.loc[resources[resources[resources.columns[-2:]] == row[-2:].values][resources.columns[-2:]].dropna().index]
    d1 = len(df)
    df = df[df.outcome == 1]
    d2 = len(df)

    d = d2/d1

    b = Counter(row)[resources.columns[-1]]
    a = Counter(resources.values.flatten())[resources.columns[-1]]
    c = Counter(data.outcome.values)[row[0]] / len(data[data[data.columns[-1]] == data.values[i, -1]])

    result.append([a, b, c, d])

  result = pd.DataFrame(result, columns=columns)


  return result


# In[13]:


a = process_data(df, 3)


# In[11]:


b = augment_data(a)


# In[ ]:


c


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(c[c.columns[2:]].values, c[c.columns[1]].values.reshape(-1, 1), test_size=0.3, random_state=1234)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

dt_pipe = Pipeline([("scaler", StandardScaler()), ("dt", DecisionTreeRegressor(max_depth=5, random_state=1234))])
gbr_pipe = Pipeline([("scaler", StandardScaler()), ("gbr", GradientBoostingRegressor(max_depth=3, random_state=1234))])


# In[ ]:


dt_pipe.fit(x_train, y_train)
gbr_pipe.fit(x_train, y_train)


# In[ ]:


dt_pipe.score(x_test, y_test)


# In[ ]:


gbr_pipe.score(x_test, y_test)


# In[ ]:
