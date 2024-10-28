import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

X = np.linspace(-10, 10, 20)
y = 2 * X + 3 + np.random.randn(20) * 3

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)

plt.plot(X, 2 * X + 3, color='r')
plt.scatter(x_train, y_train, marker='.', label='train')
plt.scatter(x_test, y_test, label='test')

plt.legend()
plt.show()