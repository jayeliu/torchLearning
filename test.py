import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
if __name__ == '__main__':
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    data=pd.read_csv("test.csv")
    for i, l_his in enumerate(data.values):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()