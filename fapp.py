# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 02:25:34 2018

@author: HP
"""

from flask import Flask, render_template,request
import pandas
from sklearn.cross_validation import train_test_split
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import tflearn
import tensorflow as tf
import random
import pickle
import sp_ask as sp_ask


app = Flask(__name__, template_folder='C:/Users/HP/Desktop/software_project/mypackage/templates')
@app.route('/')
def my_form():
    return render_template('query.html')

@app.route('/', methods=['POST'])
def my_form2_post():
    a=request.form['query']
    result = sp_ask.response(a)
    str=result[0]
    print(str[0])
    #return render_template("data.html",  data=x.to_html())
    return render_template("query.html", result=str[0])
    #return (request.form['projectFilePath'])

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)