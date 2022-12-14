from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np



def generate_confusion_matrix(df, target="", result=""):
    df[target] = df[target].apply(np.int64)
    x = df.drop([target, result], axis=1).values
    y = df[result].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # train the model
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # compute accuracy of the model
    knn.score(X_test, y_test)
    cm = confusion_matrix(y_test,y_pred)
    plt.imshow(cm, cmap='binary', interpolation='None')
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    filename = f"static/confusion_matrix_{ts}.png"
    plt.savefig(filename)
    return filename