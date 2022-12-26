from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np


app=Flask(__name__)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_Dataset.csv')

@app.route('/')
def index():
    companies=sorted(car['company'].unique())
    companies.insert(0,'Select Company')
    car_models=sorted(car['name'].unique())
    car_models.insert(0,'Select Car Models')
    year=sorted(car['year'].unique())
    year.insert(0,'Select year')
    fuel_type=sorted(car['fuel_type'].unique())
    fuel_type.insert(0,'Select Fuel Type')
    return render_template('index.html',companies=companies,car_models=car_models,years=year,fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kilo_driven'))
    print(company,car_model,year,fuel_type,kms_driven)
    prediction=model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    return str(np.round(prediction[0],2))

if __name__=='__main__':
    app.run(debug=True)