import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# Linear Regression Code by @Ronojoy Bhaumik
X = tf.constant(range(10), dtype=tf.float32)
Y = 2 * X + 10
print("X:{}".format(X))
print("Y:{}".format(Y))

X_test = tf.constant(range(10, 20), dtype=tf.float32)
Y_test = 2 * X_test + 10
print("X_test:{}".format(X_test))
print("Y_test:{}".format(Y_test))


# Define Loss Function
def loss_mse(X, Y, w0, w1):
    Y_hat = w0 * X + w1
    errors = (Y_hat - Y) ** 2
    return tf.reduce_mean(errors)


# Once you have loss( RMSE) , define derivatives using tf
def compute_gradients(X, Y, w0, w1):
    with tf.GradientTape() as tape:
        loss = loss_mse(X, Y, w0, w1)
    # loss gradient descent needs loss and get dw0 and dw1 wrt to loss
    return tape.gradient(loss, [w0, w1])


def derivative_demo():
    x = tf.constant(4.0)
    with tf.GradientTape() as g:
        g.watch(x)
        y = x * x
    dx_dy = g.gradient(y, x)  # Will compute to 8.0
    # derivative of 3 exp 2 (9)is 2*3 = 6
    print("Derivative value is: ", dx_dy.numpy())



derivative_demo()
w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)
dw0, dw1 = compute_gradients(X, Y, w0, w1)
print("dw0:", dw0.numpy())
print("dw1:", dw1.numpy())

# Run Linear Regression to minimize error
STEPS = 1000
LEARNING_RATE = .02
MSG = "STEP {step} - loss: {loss}, w0: {w0}, w1: {w1}\n"

# initialize weights
w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

# repeat a 1000 times, get derivative wrt loss and multiply with weight to adjust
# Multiplying original weight by derivative is gradient descent up or down to get to loss 0
# moving up or down to minimize loss, as derivative is wrt loss func.

for step in range(0, STEPS + 1):

    dw0, dw1 = compute_gradients(X, Y, w0, w1)
    w0.assign_sub(dw0 * LEARNING_RATE)
    w1.assign_sub(dw1 * LEARNING_RATE)
    # WITH EVERY STEP WE WE MULTIPLY DERIVATIVE( MOVEMENT) IN UP OR DOWN DIRECTION BY 0.2 TO MINIMIZE LOSS.
    # WHEN dW0/dLOSS AND dW1/dLOSS gets closer to coordinates(0,0), loss approaches 0
    if step % 100 == 0:
        loss = loss_mse(X, Y, w0, w1)
        print(MSG.format(step=step, loss=loss, w0=w0.numpy(), w1=w1.numpy()))

X = tf.constant(np.linspace(0, 2, 10), dtype=tf.float32)
print(X)
# linspace returns evenly spaced numbers over a specified interval
Y = X * tf.exp(-X ** 2)
print(Y)
plt.plot(X, Y, color='red')

plt.title('Function plot')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()

