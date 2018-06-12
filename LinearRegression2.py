import tensorflow as tf
x_train=tf.placeholder(tf.float32,shape=[None])
y_train=tf.placeholder(tf.float32,shape=[None])
W=tf.Variable(tf.random_normal([1]),name="weight")
b=tf.Variable(tf.random_normal([1]),name="bias")
hypothesis=x_train*W+b   
cost=tf.reduce_mean(tf.square(hypothesis-y_train)) 
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
  coat_val,W_val,b_val,_=sess.run([cost,W,b,train],feed_dict={x_train:[1,2,3],y_train:[1,2,3]})
  if step%20==0:
    print(step,coat_val,W_val,b_val)