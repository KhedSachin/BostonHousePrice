from flask import Flask, request, render_template,request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/predict')
def hello():
    return render_template('predict.html')
    
@app.route('/predict', methods=['post'])
def predict():
    crim = float(request.form.get('crim'))
    zn = float(request.form.get('zn'))
    indus = float(request.form.get('indus'))
    nox = float(request.form.get('nox'))
    
    rm = float(request.form.get('rm'))
    age = float(request.form.get('age'))
    dis = float(request.form.get('dis'))
    tax = float(request.form.get('tax'))
    
    ptratio = float(request.form.get('ptratio'))
    b = float(request.form.get('b'))
    lstat = float(request.form.get('lstat'))
    
    
    
    input_data = np.asarray([crim,zn,indus,nox,rm,age,dis,tax,ptratio,b,lstat]).reshape(1,-1)
    
    price = model.predict(input_data)
    output = str(round(price[0],2))
    

    return render_template('predict.html', predicted_score = f'Expected Price : ${output}')







if __name__ == '__main__':
    app.run(debug=True)