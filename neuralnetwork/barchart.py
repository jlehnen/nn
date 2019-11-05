import matplotlib.pyplot as plt
import numpy as np


def plot_activations(activations):
    fig, axes = plt.subplots(2, 4)
    for i in range(len(activations)):
        y_pos = np.arange(len(activations[i]))
        a = 0 if i < 4 else 1
        b = i % 4
        axes[a, b].barh(y_pos, activations[i], align='center')
        axes[a, b].set_yticks(y_pos)
        axes[a, b].set_yticklabels(y_pos + 1)
        # axes[a, b].set_xlabel('Activation')
        axes[a, b].set_ylabel('Neuron')
        axes[a, b].set_title('Example {}'.format(i + 1))
        axes[a, b].set_xlim(0, 1)

    fig.set_size_inches(10, 6)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle('Activations')
    fig.savefig('activations.pdf', format="pdf", bbox_inches='tight')
    plt.show()
