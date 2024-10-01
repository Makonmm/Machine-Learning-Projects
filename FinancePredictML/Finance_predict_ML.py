import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('FinancePredictML/dataset.csv')
print("\n")
print("Data was successfully loaded!")
print("Shape: ", df.shape)
print(df.head())

df.plot(x='Investment', y='Return', style='o')
plt.title('Investment x Return')
plt.xlabel('Investment')
plt.ylabel('Return')
plt.savefig('FinancePredictML/images/graph1.png')
plt.show()

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

X_train = X_train.reshape(-1, 1).astype(np.float32)

# Training

model = LinearRegression()

model.fit(X_train, y_train)
print("\n")
print("Model trained!")

print("\n")
print('B1 (coef_) :', model.coef_)
print('BO (intercept_) : ', model.intercept_)

# Ploting

regression_line = model.coef_ * X + model.intercept_
plt.scatter(X, y)
plt.title('Investment x Return')
plt.xlabel('Investment')
plt.ylabel('Predicted return')
plt.plot(X, regression_line, color='red')
plt.savefig('FinancePredictML/images/regression_line.png')
plt.show()

y_pred = model.predict(X_test)

df_values = pd.DataFrame({'Real value': y_test, 'Predicted value': y_pred})
print("\n")
print(df_values)

fig, ax = plt.subplots()
index = np.arange(len(X_test))
bar_width = 0.35
current = plt.bar(index, df_values['Real value'],
                  bar_width, label='Real value')
predicted = plt.bar(
    index + bar_width, df_values['Predicted value'], bar_width, label='Predicted Value')

plt.xlabel('Investment')
plt.ylabel('Predicted return')
plt.title('Real value x Predicted value')
plt.xticks(index + bar_width, X_test)
plt.legend()
plt.savefig('FinancePredictML/images/predicted_values.png')
plt.show()

# Evaluation

print("\n")
print('MAE (Mean Absolute Error): ', mean_absolute_error(y_test, y_pred))
print('MSE (Mean Squared Error): ', math.sqrt(
    mean_squared_error(y_test, y_pred)))
print('R2 Score: ', r2_score(y_test, y_pred))


# Predicting return with new data
print("\n")
input_inv = input("\nHow much you want to invest? ")
input_inv = float(input_inv)
inv = np.array([input_inv])
inv = inv.reshape(-1, 1)

pred_score = model.predict(inv)

print("\n")
print("Investment = ", input_inv)
print("Predicted return = {:.4}".format(pred_score[0]))
print("\n")
