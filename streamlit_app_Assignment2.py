# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:26:43 2024

@author: Sneha Oram
"""
import numpy as np
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


#pickle_in = open('parameters.pkl', 'rb')
#param = pickle.load(pickle_in)




param ={}
param["W"] = np.array([[ 0.64471707, -0.83021671, -1.56965126, -0.9094121 ,  1.67792237,
          -1.92840973, -0.40959369, -0.79055402,  2.01067215]])
param["bias"] = np.array([[0.91789416]])
param["W_feedb"] = np.array([[0.31102746]])

#from PIL import image
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(W, bias, W_feedb, X, max_sen_len):
   yhat = np.zeros((len(X), max_sen_len))
   sent_dim = len(X[0,:,0])

   for i in range(sent_dim):
      if i == 0:
         yhat_prev = np.zeros((len(X),1))
         yhat = (sigmoid(np.dot(W,X[:,i,:].T) + np.dot(W_feedb,yhat_prev.T) + bias)).T
      else:
         yhat_prev = sigmoid( np.dot( W , X[:,i,:].T ) + np.dot(W_feedb,yhat_prev.T) + bias).T
         yhat = np.concatenate((yhat, yhat_prev), axis =1)
         print(yhat.shape)

   return yhat

def predict(W, bias, W_feedb, X_train_act, max_sen_len):
    o = []
    for i in range(len(X_train_act)):
        ip = np.array([X_train_act[i]])
        op = forward(W, W_feedb, bias, ip,max_sen_len )
        y = np.where(op <= 0.5, 0, 1).tolist()[0]
        o.append(y)
        
        
    return o[-1:]

def data_processing( data_postags):
  data_postags_final = []
  Org_data_dim = []
  for i in range(len(data_postags)):
      sentence = []
      prev_word_oh = [1,0,0,0,0]
      Org_data_dim.append(len(data_postags[i]))


      for j in range(len(data_postags[i])):
          word_oh = [0]*5
          word_oh[data_postags[i][j]] = 1
          print(data_postags[i][j])
          sentence.append(prev_word_oh + word_oh[1:])
          prev_word_oh = word_oh

      data_postags_final.append(sentence)
      print(data_postags_final)

  return data_postags_final

def postags(sent):
    token = word_tokenize(sent)
    token = [i.lower() for i in token]
    text_pos = pos_tag(token)
    POStag = {}
    POStag["NN"] = 1
    POStag["JJ"] = 3
    POStag["DT"] = 2
    POStag["OT"] = 4
    l = []
    a = []
    tags = []
    for i in text_pos:
        tags.append(i[1])
    print(tags)
    for i in tags:
      if i in ["JJR", "JJS"]:
        i = "JJ"
      if i in ["NNP", "NNS"]:
        i = "NN"
      if i not in ["NN", "JJ", "DT", "JJR", "JJS", "NNP", "NNS"]:
        i = "OT"
      a.append(POStag[i])
    print(a)
    
    l.append(a)
    return l

def show_output(sent, param):
    data_postags = postags(sent)
    max_sen_len = len(data_postags[0])
    data_postags_final = data_processing( data_postags)
    output = predict(param["W"], param["bias"], param["W_feedb"], data_postags_final, max_sen_len)
    return output
    
    
    

def main():
    st.title('CS772: Assignment_2_Training of recurrent neural perceptron')
    #st.image('Screenshot_2024-02-13_232956.png', caption='Neural Network Architecture')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Noun Chunk Detector</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    #for i in range(4):
      #st.write(i+1, " fold accuracy (#inputs = 256): ", acc1[i])

    #st.write("Whole dataset accuracy (#inputs = 1024): ", acc)

    #st.write("Following results are final weights and biases")
    #st.write(param)
    
    
    input_sent = st.chat_input("Enter Text: ")
    if input_sent:
        
        output_chunk = show_output(input_sent, param)
        token = word_tokenize(input_sent)
        result = ""
        for i in range(len(token)):
            result += token[i]+"_"+ str(output_chunk[0][i]) + " "
        
        st.write(result)
        
    else:
        print("Invalid Input")
        



if __name__ == '__main__':
    main()