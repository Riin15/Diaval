'''
    Contoh Deloyment untuk Domain Data Science (DS)
    Orbit Future Academy - AI Mastery - KM Batch 3
    Tim Deployment
    2022
'''

# =[Modules dan Packages]========================

from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from joblib import load

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]   
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]   
@app.route("/predict",methods=['POST'])
def diavalpred():
    if request.method=='POST':
        # Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
        input_carat = float(request.form['carat'])
        input_cut = request.form['cut']
        input_color = request.form['color']
        input_clarity = request.form['clarity']
        input_depth = float(request.form['depth'])
        input_table = float(request.form['table'])
        input_x = float(request.form['x'])
        input_y = float(request.form['y'])
        input_z = float(request.form['z'])

        cut_mapping = {'Ideal': 0, 'Premium': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4}
        map_cut = cut_mapping.get(input_cut, -1)
        cut_color = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
        map_color = cut_color.get(input_color, -1)
        cut_clarity = {'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7}
        map_clarity = cut_clarity.get(input_clarity,-1)
        

        # Prediksi kelas atau spesies bunga iris berdasarkan data pengukuran yg diberikan pengguna
        df_test = pd.DataFrame(data={
            "carat" : [input_carat],
            "cut"  : [map_cut],
            "color" : [map_color],
            "clarity" : [map_clarity],
            "depth" : [input_depth],
            "table" : [input_table],
            "x" : [input_x],
            "y" : [input_y],
            "z"  : [input_z]
        })

        hasil_prediksi = model.predict(df_test[0:1])[0]

        return render_template('index.html', prediction=hasil_prediksi)

# =[Main]========================================

if __name__ == '__main__':
    
    # Load model yang telah ditraining
    model = load('model_diaval.model')

    # Run Flask di localhost 
    app.run(host="localhost", port=5000, debug=True)
    
    


