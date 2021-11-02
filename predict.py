import pickle
import pandas as pd
from prep import prep_df
from flask import Flask
from flask import request
from flask import jsonify

model_file='model_chosen.bin'

with open(model_file,'rb') as f_in:
    model,cat,num=pickle.load(f_in)

app = Flask('predict_fifa_overallrating') #ping is name of flask app

#put decorator on function ,will
#allow to change function to web service

@app.route('/predict',methods=['POST'])# 'POST' because we want send some information about customer
def predict():

    player=request.get_json() 
    player=prep_df( pd.DataFrame( [player] ) )
    y_pred=model.predict(player[cat+num])[0].round()
    # print(y_pred)
    
    result={
        'overall_rating:':float(y_pred)
    }
    return jsonify(result)


if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)


