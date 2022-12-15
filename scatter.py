import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def generate_scatter(dataset, x, y):
    rng = np.random.RandomState(0)
    colors = rng.rand(len(dataset))
    sizes = 1000 * rng.rand(len(dataset))
    X = dataset[x]
    Y = dataset[y]
    plt.scatter(X, Y, c=colors, s=sizes, alpha=0.3,
                cmap='viridis')
    plt.colorbar()
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    filename = f"static/scatter_{ts}.png"
    plt.savefig(filename)
    plt.close()
    return filename
