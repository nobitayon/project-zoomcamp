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


def prep_df(df):

    # Data Preparation



    df.columns=df.columns.str.lower().str.replace(' ','_')

    # df=df.drop(["unnamed:_0",'id','joined','name','nationality','photo',
    #             'flag','club','club_logo','real_face','jersey_number',
    #             'loaned_from','contract_valid_until',
    #             'potential','preferred_foot','body_type',
    #             'nationality','height','weight'],axis=1)

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


    # End data preparation

    # Split dataset

    categorical=['international_reputation','att_wr','def_wr','no_position','weak_foot','skill_moves','position']
    numerical=[col for col in df.columns if col not in categorical+['overall']]

    #transform value,wage,and release value



    df['value']=df['value'].apply(np.log1p)
    df['wage']=df['wage'].apply(np.log1p)
    df['release_clause']=df['release_clause'].apply(np.log1p)

    return df


