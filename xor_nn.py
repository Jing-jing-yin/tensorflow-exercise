import tensorflow as tf
import numpy as np
#tf.set_random_seed(777)
x_data=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data=np.array([[0],[1],[1],[0]],dtype=np.float32)

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

learning_rate=0.1

w1=tf.Variable(tf.random_normal([2,2]),name="weight")
b1=tf.Variable(tf.random_normal([2]),name="bias")
L1=tf.sigmoid(tf.matmul(x,w1)+b1)

w2=tf.Variable(tf.random_normal([2,1]),name="weight")
b2=tf.Variable(tf.random_normal([1]),name="bias")
hypothesis=tf.sigmoid(tf.matmul(L1,w2)+b2)
cost=-tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(10001):
    sess.run(optimizer,feed_dict={x:x_data,y:y_data})
    if step%100==0:
      print(step,sess.run(cost,feed_dict={x: x_data,y :y_data}),sess.run([w1,w2]))
  h,c,a=sess.run([hypothesis,predicted,accuracy],feed_dict={x: x_data,y:y_data})
  print("\nHypothesis: ",h,"\nCorrect (Y): ",c,"\nAccuracy: ",a)
