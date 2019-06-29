
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import random
import tensorflow as tf
import matplotlib.patches as mpatches
from sklearn.preprocessing import OneHotEncoder 

plt.rcParams['figure.figsize'] = (10, 6)
# 1)Creare due gruppi di dati assegnandoli come label positivo o negativo:
# I due gruppi li costurisco separadoli già con una retta nota che è y=4x +12
# in modo da vedere dopo la differenza tra la retta con cui gli ho separati in fase di creazione
# e quella con cui li separerà tensorflow
nSample = 200
alpha, epochs = 0.0035,500
feature = []
label = []


for i in range(round(nSample/2)):
    label.append([0])
    x = random.random()*6 -3
    y =  x*9+ random.random()*5000 -100
    feature.append([x,y])
  
for i in range(round(nSample/2)):
    label.append([1])
    x = random.random()*6 +3
    y = 40*x - random.random()*5000 +100
    feature.append([x,y])
    
feature=np.array(feature)
label=np.array(label)

x_pos = np.array([feature[i] for i in range(nSample) if label[i] == 1]) 
plt.scatter(x_pos[:, 0], x_pos[:, 1], color = 'red', label = 'Negative')
print("Classe positiva")
print(x_pos)
x_neg = np.array([feature[i] for i in range(nSample) if label[i] == 0]) 
plt.scatter(x_neg[:, 0], x_neg[:, 1], color = 'blue', label = 'Positive')
print("Classe negativa")
print (x_neg)
print('Dimensione del vettore delle feature: ',feature.shape)
print('Dimensione del vettore delle categorie: ',label.shape)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2') 
plt.title('Feature space') 
plt.legend() 
plt.show()
# Encoding dei dati 
oneHot = OneHotEncoder(categories='auto')
oneHot.fit(feature) 
x = oneHot.transform(feature).toarray()
oneHot.fit(label) 
y = oneHot.transform(label).toarray() 

m, n = x.shape 
print('m =', m) 
print('n =', n) 
print('Learning Rate =', alpha) 
print('Epoche =', epochs) 
# Creo variabili e placeholder per descrivere il modello 
X = tf.placeholder(tf.float32, [None, n])
Y = tf.placeholder(tf.float32, [None, 2])
W = tf.Variable(tf.zeros([n, 2]))
b = tf.Variable(tf.zeros([2]))
Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
print('out dim =', Y_hat)
# Costo da minimizzare

cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = Y)
# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer( 
         learning_rate = alpha).minimize(cost) 
# Inizializzatore globale delle variabili  
init = tf.global_variables_initializer()
diff=1
# Facciamo partire la sessione
with tf.Session() as sess: 
    sess.run(init) 
    cost_history, accuracy_history = [], [] 
    for epoch in range(epochs): 
        cost_per_epoch = 0     
        sess.run(optimizer, feed_dict = {X : x, Y : y})
        c = sess.run(cost, feed_dict = {X : x, Y : y}) 
        correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        cost_history.append(sum(sum(c))) 
        accuracy_history.append(accuracy.eval({X : x, Y : y}) * 100)
        if epoch % 100 == 0 and epoch != 0: 
            print("Epoch " + str(epoch) + " Cost: " + str(cost_history[-1])) 
    Weight= sess.run(W) 
    Bias = sess.run(b)   
    correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,  tf.float32))  
                                             
plt.plot(list(range(epochs)), cost_history) 
plt.xlabel('Epochs') 
plt.ylabel('Cost') 
plt.title('Decrease in Cost with Epochs') 
plt.show() 
plt.plot(list(range(epochs)), accuracy_history) 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.title('Increase in Accuracy with Epochs')  
plt.show()
# calcolo della retta di decisione
decision_boundary_x = np.array([np.min(feature[:, 0]), np.max(feature[:, 0])]) 
decision_boundary_y = (1.0/Weight[0])*(decision_boundary_x * Weight +Bias)
decision_boundary_y = [sum(decision_boundary_y[:, 0]), sum(decision_boundary_y[:, 1])] 
# Dati positivi
x_pos = np.array([feature[i] for i in range(len(feature))if label[i] == 1]) 
# Dati negativi
x_neg = np.array([feature[i] for i in range(len(feature)) if label[i] == 0]) 
# Plotto i due cluster 
plt.scatter(x_pos[:, 0], x_pos[:, 1], color = 'red', label = 'Negative') 
plt.scatter(x_neg[:, 0], x_neg[:, 1], color = 'blue', label = 'Positive') 
# Plotto la retta di decisione
plt.plot(decision_boundary_x,decision_boundary_y)
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.title('Plot of Decision Boundary') 
plt.legend() 
plt.show() 
