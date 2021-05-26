import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy import stats
import xgboost as xgb

# load training data, removing the split column
cols = list(pd.read_csv("/train.csv", nrows =1))
data = pd.read_csv("/train.csv", usecols =[i for i in cols if i != 'split']
                   ,na_values = [' ', '.'])
# keep only complete cases
df_train0 = data.dropna()

#select feature
df_train = df_train0.iloc[:,9:]

# load test data, removing the split column
cols1 = list(pd.read_csv("/participant_test.csv", nrows =1))
df_test = pd.read_csv("/participant_test.csv", usecols =[i for i in
cols1 if i not in['UNIQUE_ID','split'] ]
                   ,na_values = [' ', '.'])
all_data = pd.concat([df_train,df_test], axis=0)

#create dummy variables for certain categorial features
cat_columns1 = ['SJ_Most_1', 'SJ_Least_1', 'SJ_Most_2', 'SJ_Least_2',
'SJ_Most_3', 'SJ_Least_3', 'SJ_Most_4', 'SJ_Least_4',  'SJ_Most_5',
'SJ_Least_5',  'SJ_Most_6', 'SJ_Least_6',  'SJ_Most_7', 'SJ_Least_7',
'SJ_Most_8', 'SJ_Least_8', 'SJ_Most_9', 'SJ_Least_9',
               'Scenario1_1', 'Scenario1_2', 'Scenario1_3',
'Scenario1_4', 'Scenario1_5', 'Scenario1_6', 'Scenario1_7',
'Scenario1_8',  'Scenario2_1', 'Scenario2_2', 'Scenario2_3',
'Scenario2_4', 'Scenario2_5', 'Scenario2_6', 'Scenario2_7',
'Scenario2_8',
               'Biodata_01', 'Biodata_02', 'Biodata_03', 'Biodata_04',
'Biodata_05', 'Biodata_06', 'Biodata_07',
               'Biodata_08', 'Biodata_09', 'Biodata_10', 'Biodata_11',
'Biodata_12', 'Biodata_13', 'Biodata_14',
               'Biodata_15', 'Biodata_16', 'Biodata_17', 'Biodata_18',
'Biodata_19', 'Biodata_20']
all_data = pd.get_dummies(all_data, prefix_sep="__",columns=cat_columns1)
X_train  = all_data[:df_train.shape[0]]
X_test   = all_data[-df_test.shape[0]:]

ID_test = pd.read_csv("/participant_test.csv", usecols =['UNIQUE_ID'])


#high_performer prediction; with default hyperparameters; convert the prediction scores into z scores
y_train= df_train0['High_Performer']
model = XGBClassifier(scale_pos_weight=1.5)
model.fit(X_train, y_train)
# make predictions for test data and convert into z scores
y_pred1_mod = model.predict_proba(X_test)
y_pred1_prob = y_pred1_mod[:,1]
y_pred1=stats.zscore(y_pred1_prob)


#Overall_rating prediction
y_train= df_train0['Overall_Rating']
xg_reg = xgb.XGBRegressor( subsample=0.6, n_estimators=200,
                          learning_rate= 0.05,
                          colsample_bytree= 0.6, max_depth=2,
                           min_child_weight=4)
xg_reg.fit(X_train, y_train)
y_pred2_score = xg_reg.predict(X_test)
y_pred2=stats.zscore(y_pred2_score)

####fine turn hyper-parameters for high_performer prediction
####random forest for Bayesian optimization was conducted
y_train= df_train0['High_Performer']
model = xgb.XGBClassifier(learning_rate = 0.005411872947900535,
                            n_estimators = 1789,
                            max_depth = 3,
                            min_child_weight = 5.818676232053935,
                            gamma = 0.05591980172280099,
                            subsample = 0.5744551033482959,
                            colsample_bytree = 0.5226781217635239, seed = 42)

model.fit(X_train, y_train)
# make predictions for test data
y_pred3_mod = model.predict_proba(X_test)
y_pred3_prob = y_pred3_mod[:,1]
y_pred3=stats.zscore(y_pred3_prob)

#### protected and retained worker predictions;
def double(r):
  if r.Protected_Group == 1 and r.Retained==1:
    return 1
  else:
    return 0

y_train = df_train0.apply(double, axis=1)
model = XGBClassifier(scale_pos_weight=3)
model.fit(X_train, y_train)
# make predictions for test data
y_pred4_mod = model.predict_proba(X_test)
y_pred4_prob = y_pred4_mod[:,1]
y_pred4=stats.zscore(y_pred4_prob)

###final weighted ensemble model
###predict top performers then adjusted by minority & retained workers
y_pred= (y_pred1*.5+y_pred2*.2+y_pred3*.3)*.9+y_pred4*.1
y_pred= pd.DataFrame(y_pred)
#select the top half subgroup to "hire"
cut_off=np.median(y_pred.iloc[:,0])
y_pred['Hire'] = np.where(y_pred.iloc[:,0]> cut_off , 1, 0)

#save the results to csv file
sub = pd.DataFrame()
sub['UNIQUE_ID']=ID_test['UNIQUE_ID']
sub['Hire']=y_pred['Hire']
sub.to_csv('final_submission.csv',index=False)
