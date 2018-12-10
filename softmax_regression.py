import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32, [None,10])

# 构造交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
# 梯度下降优化损失  0.01为学习率
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建一个session
sess = tf.InteractiveSession()
# 运行之前对要初始化所有的变量，分配内存
tf.global_variables_initializer().run()

# 有了会话就可以对变量进行优化了，优化的程序为：
# 每次不使用全部的数据，而是选取100个数据进行训练，训练1000次
for _ in range(1000):
    # 在mnist.train中取100个训练数据
    # batch_xs是（100,784），batch_ys是（100，10）
    batch_xs,batch_ys = mnist.train.next_batch(100)
    # 在session中运行train_step，运行时要传入占位符的值
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

# 通过equal函数比较实际值和计算出的值，返回True or False
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# 通过cast函数将True返回为1，False返回为0
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# 在Session中运行tensor可以得到tensor的值
# 获取最终模型的准确率
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


