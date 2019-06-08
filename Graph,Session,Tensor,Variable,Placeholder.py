import tensorflow as tf
graph1 =tf.Graph()
# a and b are two one-dimensional tensors
with graph1.as_default():
    a = tf.constant([2], name = 'constant_a')
    b = tf.constant([3], name = 'constant_b')
# for see tensors
sess=tf.Session(graph=graph1) 
result=sess.run(a)
print(result)
result=sess.run(b)
print(result)
# summ of tensors
c=tf.add(a,b)
c=a+b
sess.close
# v is a 3-dimensional array
graph2 = tf.Graph()
with graph2.as_default():
    v = tf.constant([5,6,7])
# m is a 3*3-dimensional matrix
    m = tf.constant([[5,26,7],[5,65,7],[45,6,77]])
# t is a 3*3*2tensor
    t= tf.constant([[[51,2,7],[5,5,7],[45,68,77]],[[5,26,7],[5,65,7],[45,6,77]]])
with tf.Session(graph=graph2) as sess:
    result=sess.run(v)
    print("v \n %s"%result)
    result=sess.run(m)
    print("m \n %s"%result)
    result=sess.run(t)
    print("t \n %s"%result)
#difference beetween tensorflow add e normal add is that tensorflow operations are nodes that represent the mathematical operations over the tensors on a graph
graph3 = tf.Graph()
with graph3.as_default():
    Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

    add_1_operation = tf.add(Matrix_one, Matrix_two)
    add_2_operation = Matrix_one + Matrix_two

with tf.Session(graph =graph3) as sess:
    result = sess.run(add_1_operation)
    print ("Defined using tensorflow function :")
    print(result)
    result = sess.run(add_2_operation)
    print ("Defined using normal expressions :")
    print(result)
#hadamard product
graph4 = tf.Graph()
with graph4.as_default():
    Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])
    matmuloperation = tf.matmul(Matrix_one, Matrix_two)
with tf.Session(graph =graph3) as sess:
    result = sess.run(add_1_operation)
    print ("Defined using tensorflow function :")
    print(result)

#Variables, the differnce betwen variables and tensor are in the persistence. Variables are persistent in different session
v= tf.Variable(0)
init_op=tf.global_variables_initializer()
update= tf.assign(v,v+1)
with tf.Session()as session:
    session.run(init_op)
    print(session.run(v))
    for i  in range(3):
        session.run(update)
        print(session.run(v))
        
#Placeholder, we can define a simple operation b=a*2 if a is a placeholder we can( and we are obbligated)
#to pass an argument with b for run b in session
#define placeholder
a = tf.placeholder(tf.float32)
#define operation
b = a*2
with tf.Session() as sess:
    result=sess.run(b,feed_dict={a:3.5})
    print (result)
#To pass the data into the model we use a dict. This means that we can define the dict out of the session
dictionary={a:5}
with tf.Session() as sess:
    result=sess.run(b,feed_dict=dictionary)
    print(result)
