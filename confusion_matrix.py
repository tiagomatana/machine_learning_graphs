from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


def generate_confusion_matrix(df, result=""):
    X = df.drop([result], axis=1).values
    y = df[result].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
    #
    # Create the pipeline
    #
    pipeline = make_pipeline(StandardScaler(),
                             RandomForestClassifier(n_estimators=10, max_features=5, max_depth=2, random_state=1))
    #
    # Fit the Pipeline estimator
    #
    pipeline.fit(X_train, y_train)
    #
    # Get the predictions
    #
    y_pred = pipeline.predict(X_test)
    #
    # Calculate the confusion matrix
    #
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    #
    # Print the confusion matrix using Matplotlib
    #
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    filename = f"static/confusion_matrix_{ts}.png"
    plt.savefig(filename)
    plt.close()
    return filename