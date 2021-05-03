from flask import Flask, render_template, request,jsonify
#Flask web için python frameworku
#render_template ile birlikte kullanılan html renderlanır
#flask ile datamızı json haline getirmeye jsonify denir
from keras.models import load_model # model yüklemek için kullanılır
import cv2 #gorsel işlemler için
import numpy as np #dizi,matrix işlemlerimiz için kullanırız
import base64 #encode işlemlerinde kullanılır
from PIL import Image #gelen image
import io
import re
import os
img_size=100
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices[0])
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#gpunun memory alanı kullanmasına izin veriyorsun
app = Flask(__name__) #su anki kullandıgımız modein ismi verilir app.py
export_path = os.path.join(os.getcwd(), 'model', 'model-012.model')
model=load_model(export_path) #olusturgumuz modellerden model-012 secildi

label_dict={0:'Covid19 Negative', 1:'Covid19 Positive'}
#negative 0 positive 0 olarak labellandı
def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #griye cevirildi
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=resized.reshape(1,img_size,img_size)
	return reshaped #aldıgımız test image train seklinde image sekline cevrilir

@app.route("/") #ilk cagrılma gereken url bizim htmlimiz olacak
def index():
	return(render_template("index.html"))
     #olusturdugumuz htmli template olarak alırız
@app.route("/predict", methods=["POST"]) #predict url cagrıldıgı zaman gelen image alınacak
def predict():
	print('HERE')
	message = request.get_json(force=True) #jsondan gelen mesaj datası alınır
	encoded = message['image'] #mesajdan image alınır
	decoded = base64.b64decode(encoded) #gelen sifreli mesaj decodelanır
	dataBytesIO=io.BytesIO(decoded) #data byte olarak tutulur
	dataBytesIO.seek(0) #datanın basına donuldu
	image = Image.open(dataBytesIO) #data stream okunur ve image alınır

	test_image=preprocess(image) #alınan resim preprocess fonksiyonuna gonderilir ve 1,100,100 haline getirilir

	prediction = model.predict(test_image) #modelde prediction yapılır
	result=np.argmax(prediction,axis=1)[0] #sonuc alınır
	accuracy=float(np.max(prediction,axis=1)[0]) #yuzdesi alınır

	label=label_dict[result] #alınan result 0 ise covid negative 1 ise covid positive basılacaktır.

	print(prediction,result,accuracy) 

	response = {'prediction': {'result': label,'accuracy': accuracy}} #elimizde olan degerler response olarak birleştirilir

	return jsonify(response) #jsona cevrilip gonderilir.

app.run(host='0.0.0.0', port=5000) #olusturdugumuz modülü calıstırma 

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">