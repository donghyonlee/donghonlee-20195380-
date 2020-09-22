import pandas as pd
import datetime
import numpy as np

data = pd.read_csv("C:/Users/admin/Desktop/small3insert.csv")
df = pd.DataFrame(data)
df = df.rename(columns={'Case':'caseid','Activity':'activity','Timestamp':'ts','Resource' : 'resource','resource_anomaly_type' : 'outcome'})
df = df.loc[:,['caseid','activity','ts','resource','outcome']]
#df2 = df2.rename(columns={'Case ID':'caseid','Activity':'activity','Complete Timestamp':'ts','Resource':'resource','(case) Accepted':'outcome'}
#df2 = df2.loc[:,['caseid','activity','ts','resource','outcome']]

##df = df.groupby(['activity','resource']).apply(lambda x : [x.value_counts().to_dict()]).str[0].reset_index()
##df['freq'] = df.groupby(['activity','resource']).activity.value_counts(normalize=True).reset_index(level = 0, drop = True)

re_group = df.groupby('caseid')
aggregation_encode = []
##for case,group in re_group:
     ##group = group.reset_index(drop=True)
     ##outcome = set(group['outcome']).pop()
     ##durationlist = [(i - list(group['ts'])[0]).total_seconds() for i in list(group['ts'])]
     ## NEW_TS = np.mean(durationlist)
     ##case_time_outcome = {'caseid':case, 'ts': NEW_TS,'outcome':outcome}
     # activity_count = {i: list(group['activity']).count(i) for i in set(group['activity'])}
     # resource_count = {i: list(group['resource']).count(i) for i in set(group['resource'])}

#4
 # group = df.groupby('caseid')
 # prefix_len[]
 # for group in re_group
 #  group = group.reset_index(drop = true)
 #  if len(group)>=prefix:
    #group = group.loc[:prefix-1,:]
    #prefix_len[].append(group)
    #prefix_len.appended(group)
#5
## dummy_df = pd.get_dummies(df,columns=['activity','resource','ts'])
##for case, group in dummy_df.groupby(')
## activitylist = list(group['activity'])
##activity_index = {i }

#6 from sklearn import datasets
#  from sklearn.model_selection import train_test_split
# x = data[['outcome','caseid']]
# y = data['outcome']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# forest = RandomForestClassifier(n_estimators=100)
# forest.fit(x_train, y_train)
# y_pred = forest.predict(x_test)

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matri
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# tree = DecisionTreeClassifier(max_depth = 5, random_state = 0)
# tree.fit(x_train,y_train)

