import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y_and = np.array([[0], [0], [0], [1]], dtype=np.float32)

model_and = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,))
])

model_and.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model_and.fit(X_and, Y_and, epochs=1000, verbose=0)

weights, bias = model_and.layers[0].get_weights()
w1, w2 = weights[0][0], weights[1][0]
b = bias[0]

plt.figure(figsize=(8, 6))

for i in range(len(X_and)):
    if Y_and[i] == 0:
        plt.scatter(X_and[i][0], X_and[i][1], color='red', label='0' if i == 0 else "")
    else:
        plt.scatter(X_and[i][0], X_and[i][1], color='blue', label='1' if i == 0 else "")

x_values = np.linspace(-0.5, 1.5, 100)
y_values = -(w1 * x_values + b) / w2

plt.plot(x_values, y_values, label='Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('AND Gate Perceptron Model')
plt.grid(True)
plt.show()