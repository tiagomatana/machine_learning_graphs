from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
import numpy as np


def generate_confusion_matrix(df, target="", labels="", base=""):
    X = df.drop([target, labels], axis=1).values
    y = df[target].values
    class_names = list(dict.fromkeys(df[labels].values))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classifier = svm.SVC(kernel="linear", C=0.01).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    response=[]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
        filename = f"static/confusion_matrix_{'normalized_' if normalize else ''  }{ts}.png"
        plt.savefig(filename)
        response.append(f"{base}{filename}")

    plt.close()
    return response