from flask import Flask,request,jsonify 
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)


@app.route('/api')
def y():
 
   if(request.method=='GET'):
      mydata={
         'temperature':[request.args['temperature']],
         'humidity':[request.args['humidity']],
         'ph':[request.args['ph']],
         'rainfall':[request.args['rainfall']]
      }
   
   print(mydata)
   predictedfile=pd.DataFrame(mydata)
   predictedfile.to_csv('testdata.csv',index=False)
   data=pd.read_csv('testdata.csv')


   model=pickle.load(open('model.pkl','rb'))


   print(data)
   predictions = model.predict(data)
   print(predictions[0])
   x=predictions[0]
   
   return str(x)

if __name__ == '__main__':
   app.run()