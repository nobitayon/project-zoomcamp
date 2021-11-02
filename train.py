import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle

def adjusted_r2(r2,df):
    return 1- ( (1-r2**2)*(len(df)-1) )/(len(df)-df.shape[1]-1)


# Parameters

list_md=list(range(5,20))
list_msl=list(range(5,200,10))
threshold=0.8
output_file='model_chosen.bin'

# Data Preparation

print('Start data preparation')

df=pd.read_csv('data.csv')
df.columns=df.columns.str.lower().str.replace(' ','_')

df=df.drop(["unnamed:_0",'id','joined','name','nationality','photo',
            'flag','club','club_logo','real_face','jersey_number',
            'loaned_from','contract_valid_until',
            'potential','preferred_foot','body_type',
            'nationality','height','weight'],axis=1)

df = df.dropna(subset=['release_clause'] )
df=df.reset_index(drop=True)

# Data Preparation - Prepare feature release_clause

df.release_clause=df.release_clause.astype('str')

df['dummy']=np.arange(len(df))
def fun0(row):
    unit=row.release_clause[-1]
    row.dummy=unit
    row.release_clause=row.release_clause.replace('€','').replace(unit,'')
    return row

df=df.apply(fun0,axis=1)
df.dummy=df.dummy.map({'M': 10**6, 'K': 10**3})

def fun1(row):
    row.release_clause=float(row.release_clause) * row.dummy
    return row

df=df.apply(fun1 ,axis=1,)

del df['dummy']

# Data Preparation - Prepare feature wage

df.wage=df.wage.astype('str')

df['dummy']=np.arange(len(df))
def fun2(row):
    unit=row.wage[-1]
    row.dummy=unit
    row.wage=row.wage.replace('€','').replace(unit,'')
    return row

df=df.apply(fun2,axis=1)
df.dummy=df.dummy.map({'M': 10**6, 'K': 10**3})

def fun3(row):
    row.wage=float(row.wage) * row.dummy
    return row

df=df.apply(fun3 ,axis=1,)

del df['dummy']

# Data preparation - prepare feature value

df.value=df.value.astype('str')

df['dummy']=np.arange(len(df))
def fun4(row):
    unit=row.value[-1]
    row.dummy=unit
    row.value=row.value.replace('€','').replace(unit,'')
    return row

df=df.apply(fun4,axis=1)
df.dummy=df.dummy.map({'M': 10**6, 'K': 10**3})

def fun5(row):
    row.value=float(row.value) * row.dummy
    return row

df=df.apply(fun5 ,axis=1,)

del df['dummy']

# Data preparation - prepare feature work_rate

df = df.dropna(subset=['work_rate'] )
df=df.reset_index(drop=True)
df['dummy']=df.work_rate.str.split('/')
#create column dummy , where value of each row is like ['Low','Medium'] etc ,where first element is for attacking work rate
# and second element for defending work rate

def split(row):
    row['att_wr']=row.dummy[0].replace(' ','')   #attacking work rate 
    row['def_wr']=row.dummy[1].replace(' ','')   #defending work rate
    
    return row

df.loc[:,'att_wr']=np.arange(len(df)) 
df.loc[:,'def_wr']=np.arange(len(df))
df=df.apply(split,axis='columns')

work_values = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}

df['att_wr'] = df['att_wr'].map(work_values) 
df['def_wr'] = df['def_wr'].map(work_values)

del df['work_rate']
del df['dummy']

# Data preparation - prepare feature no_position

df.position=df.position.str.lower()
def fun0(row):
    
    for pos in ['LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB']:
        pos=pos.lower()
        
        row[pos]='20+0' # Assume value of feature pos in row is 20
        
    return row

df.loc[df.position=='gk']=df.loc[df.position=='gk'].apply(fun0,axis='columns') 
#apply fun0 to row of goalkeeper because every row with position 'gk' have NaN value 

df['no_position']=np.zeros(len(df)) #create column no_position to hold value of number of position player can play

df = df.dropna(subset=['ls','st', 'rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb'] )
df=df.reset_index(drop=True)

for pos in ['ls','st', 'rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb'] :
    df['dummy'+'_'+pos]=df[pos].str.split('+') #create dummy column to hold value like ['85','2'] etc 
    df[pos]=[int( df.loc[i,'dummy'+'_'+pos][0] ) > 85 for i in range(len(df))]
    #take only the first element and convert it into integer.I also assume
    #player can play at position pos if the rating is more than 85
    
    df['no_position']=df['no_position'] + df[pos] # Add 1 if player can play in position pos
    
    del df['dummy'+'_'+pos] #delete dummy column
    
def fun1(row): #define function to add 1 to player that have rating less than 85 in his natural position
    pos=row.position.lower()
    if pos!='gk':
        if row[pos]==False:
            row['no_position']=row['no_position']+1
    elif pos=='gk':
        row['no_position']=row['no_position']+1
        
    return row

df=df.apply(fun1,axis='columns')


for pos in ['LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB']:
    pos=pos.lower()
    del df[pos]


print('done data preparation')
# End data preparation

# Split dataset

categorical=['international_reputation','att_wr','def_wr','no_position','weak_foot','skill_moves','position']
numerical=[col for col in df.columns if col not in categorical+['overall']]

df_gk=df.loc[df.position=='gk']
df_nogk=df.loc[df.position!='gk']

df_full_train_gk,df_test_gk=train_test_split(df_gk,test_size=0.2,random_state=1)

df_full_train_nogk,df_test_nogk=train_test_split(df_nogk,test_size=0.2,random_state=1)

df_full_train=pd.concat([df_full_train_gk, df_full_train_nogk],ignore_index=True)
df_test=pd.concat([df_test_gk, df_test_nogk],ignore_index=True)

#transform value,wage,and release value

df_full_train['value']=df_full_train['value'].apply(np.log1p)
df_full_train['wage']=df_full_train['wage'].apply(np.log1p)
df_full_train['release_clause']=df_full_train['release_clause'].apply(np.log1p)

df_test['value']=df_test['value'].apply(np.log1p)
df_test['wage']=df_test['wage'].apply(np.log1p)
df_test['release_clause']=df_test['release_clause'].apply(np.log1p)

# End split data set

# Function to drop feature

#This function return feature that will be dropped based on threshold
def drop_feature(dataset,threshold):
    corr_matrix=dataset.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return to_drop

# Define function to build model

def rf_pipeline(df_full_train,y_train,cat,num,list_md,list_msl):
    categorical_transformer = Pipeline(
        [
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
        ]
    )
    
    preprocessor = ColumnTransformer(
        [
            ('categoricals', categorical_transformer, cat)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(
        [
            ('preprocessing', preprocessor),
            ('rf', RandomForestRegressor())
        ]
    )
    
    param_grid=[{
    'rf__max_depth':list_md,
    'rf__min_samples_leaf':list_msl,
    'rf__random_state':[1],
    'rf__n_estimators':[10]
    }]
    
    cv_rf = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error' ,cv = 5 ,n_jobs = -1,verbose=1,return_train_score=True)
    cv_rf.fit(df_full_train,y_train)
    
    return cv_rf


# Feature selection

def mutual_info(series):
    return mutual_info_score(series,overall_categorical)

overall_categorical=pd.qcut(df_full_train.overall,q=5)
mi=df_full_train[categorical].apply(mutual_info)
categorical_th2=list(mi.sort_values(ascending=False).index[:3])

to_drop=drop_feature(df_full_train[numerical],threshold=threshold)
numerical_th2=numerical.copy()

for num in to_drop:
    numerical_th2.remove(num)

# End feature selection

# Build model
print('Start build model and search for best max_depth and min_samples_leaf from list_md and list_msl')
cv_rf=rf_pipeline(df_full_train[categorical_th2+numerical_th2],df_full_train['overall'],categorical_th2,numerical_th2,list_md,list_msl)

print('Done search')
print('')
print('Best parameter from search on list_md and list_msl')
print(cv_rf.best_params_)
print('')
print('performance on train set')
y_pred=cv_rf.predict(df_full_train[categorical_th2+numerical_th2]).round()
print(f'mse on train set : {mean_squared_error(df_full_train.overall.values,y_pred)}')
r2 = r2_score(df_full_train.overall.values,y_pred)
print(f'R2 score : {r2}')
print(f'adjusted R2 score : {adjusted_r2(r2,df_full_train)}')

print('')
print('performance on test set')
y_pred=cv_rf.predict(df_test[categorical_th2+numerical_th2]).round()

print(f'mse on test set : {mean_squared_error(df_test.overall.values,y_pred)}')

r2 = r2_score(df_test.overall.values,y_pred)
print(f'R2 score : {r2}')
print(f'adjusted R2 score : {adjusted_r2(r2,df_test)}')

#save the model
print('')

with open(output_file,'wb') as f_out:
    pickle.dump((cv_rf,categorical_th2,numerical_th2),f_out)

print(f'the model is saved to {output_file}')




