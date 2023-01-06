# Plotting of Learning Curves
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def plot_learning_curve(x, scores, ylabel, EXP_NAME, WINDOW=100):
    fig, ax = plt.subplots()
    running_avg = np.zeros(len(scores))

    pickle_path = f"{cwd}/../Pickle/{EXP_NAME}/{ylabel}"
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    z = [" ".join(str(item)) for item in zip(x, scores)]
    ofile = open(pickle_path, "wb")
    pickle.dump(z, ofile)

    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - WINDOW) : (i + 1)])
    ax.plot(x, running_avg)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    fig.suptitle(f"{EXP_NAME}: Running average for {ylabel} over {WINDOW} episodes")
    fig_path = f"{cwd}/../Plots/{EXP_NAME}/{ylabel}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)


if __name__ == "__main__":
    EXP_NAME = input("Enter experiment name: ")
    run = int(input("Enter run number: "))
    global cwd
    cwd = os.path.dirname(os.path.realpath(__file__))
    scores = np.load(f"{cwd}/../Data/{EXP_NAME}/Run_{run}.npy")
    print(scores.shape)
    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(x, scores, "Return", EXP_NAME, WINDOW=150)

    # Actor Loss
    actor_loss = np.load(
        f"{cwd}/../Data/{EXP_NAME}/Actor_loss/Actor_loss_run_{run}.npy",
        allow_pickle=True,
    )
    # actor_loss = actor_loss.reshape(1400)
    print(actor_loss, actor_loss.shape)
    x = [i + 1 for i in range(len(actor_loss))]
    plot_learning_curve(x, actor_loss, "Actor Loss", EXP_NAME, WINDOW=50)

    # Critic Loss
    critic_loss = np.load(
        f"{cwd}/../Data/{EXP_NAME}/Critic_loss/Critic_loss_run_{run}.npy"
    )
    print(critic_loss.shape)
    x = [i + 1 for i in range(len(critic_loss))]
    plot_learning_curve(x, critic_loss, "Critic Loss", EXP_NAME, WINDOW=50)

    # Show the plots
    plt.show()
