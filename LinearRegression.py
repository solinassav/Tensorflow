#Library
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import random
import tensorflow as tf
import matplotlib.patches as mpatches
plt.rcParams['figure.figsize'] = (10, 6)
#X belongs to R^n |X[i] belongs to [a,b],n=(b-a)/delta, X[i]=X[i-1]+delta
a=0.0
b=5.0
delta=0.1
X = np.arange(a,b,delta)

#Linear regression is used for the approximation of linear model like that
m = 13
q = 18
Y= m * X + q 
plt.plot(X, Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent V ariable')
plt.show()
#In the plot you can visualize a sample linear model
#Linear model are often used for the description of phisical model
#The speed in function of time is a linear model v = a*t +v0
#We can image a ball free falling from 10m. The ball touch the flow with speed=98 m/s
#This is descrivible by a linear model.
b=10.0
t = np.arange(a,b,delta)
g=-9.8
v0=0
v=g*t+v0
plt.plot(t, v) 
plt.ylabel('Speed')
plt.xlabel('Time')
plt.show()
#Now let us consider that we do not know the law that regulates motion with continuous acceleration.
#We have we have collected this values of speed whit an error of mesaurement 
error=[]
for i in range(round((b-a)/delta)):
    error.append(random.random()*2)
cv=v+np.asarray(error)
plt.plot(t, cv) 
plt.ylabel('Speed')
plt.xlabel('Time')
plt.show()
X=tf.placeholder("float")
Y=tf.placeholder("float")
W=tf.Variable(np.random.randn(),name="W")
b=tf.Variable(np.random.randn(),name="b")
learning_rate = 0.0001
training_epochs = 1000

#Now we can start the training from a casual hypothesis
y_pred =tf.add(tf.multiply(X,W),tf.cast(b,tf.float32))

#Mean squared Error function
loss= tf.reduce_mean(tf.square(y_pred-Y))

#Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#Variables initializer
init =tf.global_variables_initializer()
#Starting the session
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for(_x,_y) in zip (t,cv):
        # Remember: with variables we can pass data into a model like a dictionary
            sess.run(optimizer,feed_dict={X:_x,Y:_y})
        if(epoch+1)%50==0:
            c=sess.run(loss,feed_dict={X:t,Y:cv})
            print("Epoch", epoch+1,": loss =",c,"W =",sess.run(W), "b =", sess.run(b)) 
    training_cost=sess.run(loss,feed_dict={X:t,Y:cv})
    weight=sess.run(W)
    bias=sess.run(b)
    plt.plot(t,cv)
    plt.plot(t,weight*t+bias)
    plt.show()
              

    
    

