
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from datetime import datetime

def generate_scatter(dataset, columns):
    df = dataset
    if len(columns) > 0:
        if ',' in columns:
            columns = columns.split(',')
            df = df.drop(columns=dataset.columns.values)
            for c in columns:
                df[c] = dataset[c]
        else:
            df = dataset[columns]

    scatter_matrix(df)
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    filename = f"static/scatter_{ts}.png"
    plt.savefig(filename)
    return filename
