
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

A = np.array([1,2,3,4,10,18,12]).reshape(-1,1)
B = np.array([48, 62, 76, 79, 69, 92, 15])

A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.25, random_state = 42 )

model = LinearRegression()

model.fit(A_train, B_train)

B_pred = model.predict(A_test)

print(f"The Intercept is: {model.intercept_}")
print(f"The Slope is: {model.coef_[0]}")

print(f"The Mean Squared Error is: {mean_squared_error(B_test, B_pred)}")
print(f"The R^2 Score is: {r2_score(B_test, B_pred)}")

plt.scatter(A_train, B_train, color='red', label='Training data')
plt.scatter(A_test, B_test, color='blue', label = "Test data")
plt.plot(A, model.predict(A), color='green', linewidth=2, label='Regression line')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.title("Linear Regression Model")
plt.legend()
plt.show()
