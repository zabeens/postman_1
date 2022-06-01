import numpy as np
import pandas as pd
import json
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prob = model.predict_proba(final_features)
    term_output='{0:.{1}f}'.format(prob[0][1], 2)
    retent_output='{0:.{1}f}'.format(prob[0][0], 2)

    if term_output>str(0.5):
        prediction_text = "Attrition Probability: {}".format(term_output)
        return prediction_text 
    else:
        prediction_text = "Retention Probability: {}".format(retent_output)
        return prediction_text 
    

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    predict_request=[[data[0]['gender'],data[0]['unified_grade'],data[0]['city'],data[0]['sub_bu'],data[0]['ijp_90_days'],data[0]['tenure'],data[0]['bench_ageing'],data[0]['marital_status'],
                                data[0]['education_score'],data[0]['leaves'],data[0]['months_since_last_promotion'],data[0]['payposition_2021'],
                                data[0]['tenure_in_capgemini_yr'],data[0]['Average_Rating'],data[0]['rating_diff']]]
    req=np.array(predict_request)
    print(req)
    prediction = model.predict(req)
    prob = model.predict_proba(req)
    term_output='{0:.{1}f}'.format(prob[0][1], 2)
    pred = prediction[0]
    if term_output>str(0.5):
        Status = "Yes"        
    else:
        Status = "No"
    return jsonify("Attrition Probability : {}".format(str(term_output)),"Employee Attrite : {}".format(str(Status)))

if __name__ == "__main__":
    app.run(debug=True)
