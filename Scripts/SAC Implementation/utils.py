import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
global cwd

def plot_learning_curve(x, scores, ylabel, figure_file, WINDOW=100):
    fig, ax = plt.subplots()
    # z = [" ".join(str(item)) for item in zip(x, scores)]
    # ofile = open("plot_arr",'wb')
    # pickle.dump(z, ofile)
    cwd = os.path.dirname(os.path.realpath(__file__))
    pickle_path = f"{cwd}/../../Pickle/{'SAC_2500'}/{ylabel}"
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    z = [" ".join(str(item)) for item in zip(x, scores)]
    ofile = open(pickle_path, "wb")
    pickle.dump(z, ofile)

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-WINDOW):(i+1)])
    ax.plot(x, running_avg)
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    fig.suptitle('SAC: running average of previous 100 scores')
    plt.savefig(figure_file)
    