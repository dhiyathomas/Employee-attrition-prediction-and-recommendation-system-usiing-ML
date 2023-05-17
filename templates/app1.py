# numpy
import numpy as np
# classifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from random import shuffle
import pandas
from sklearn import model_selection, preprocessing, naive_bayes
import string
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')
@app.route('/login')
def login():
    return render_template('login.html')
def home():
	return render_template('home.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df) 

@app.route('/prediction1')
def prediction1():
    return render_template('home.html')  
@app.route('/chart')
def chart():
    return render_template('chart.html')
@app.route('/prediction')
def prediction():
 	return render_template("home.html")
@app.route('/crime')
def crime():
 	return render_template("crime.html")
@app.route('/crimes')
def crimes():
 	return render_template("crimes.html")
@app.route('/total')
def total():
 	return render_template("total.html")
@app.route('/theft')
def theft():
    return render_template('theft.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("some.csv", encoding="latin-1")
	# Features and Labels
	df['label'] = df['l']
	X = df['t']
	y = df['label']
	print(X)
	# Extract Feature With CountVectorizer
	cv = CountVectorizer(ngram_range=(1, 2))
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB

	clf = svm.SVC(kernel='linear')
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)