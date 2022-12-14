from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

def generate_linear_regression(df, x, y):
    X = df[[x]]
    Y = df[y]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    # # Criando o modelo LinearRegression
    regr = LinearRegression()
    # # Realizar treinamento do modelo
    regr.fit(X_train, y_train)
    # # Realizar predição com os dados separados para teste
    y_pred = regr.predict(X_test)
    slope = regr.coef_
    intercept = regr.intercept_

    plt.rcParams['figure.figsize'] = (11,7)
    plt.scatter(X_test, y_test,  color='b')
    plt.plot(X_test,(X_test * slope + intercept), color='r', linewidth=2)
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    filename = f"static/{ts}.png"
    plt.savefig(filename)
    return filename