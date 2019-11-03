import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from barchart import plot_activations

np.set_printoptions(linewidth=400)

L_RATE = 0.7
LAMBDA = 0.00001
EPOCHS = 10000
NUM_INPUTS = 8
NUM_HID = 3


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1. - x)


def loss(h, y):
    return (np.square(h - y)).mean()


def hyper_parameter_optimization(verbose=True, runs=10):
    # np.random.seed(42690)
    exmple = expct = np.eye(NUM_INPUTS)
    df = pd.DataFrame(columns=['l_rate', 'lambda', 'final_loss'])
    for i in tqdm(range(runs), disable=not verbose):
        for l_rate in [0.1, 0.3, 0.5, 0.7]:
            for lambd in [0.00001, 0.0001, 0.005]:
                nn = Network(NUM_INPUTS, NUM_HID, NUM_INPUTS, l_rate=l_rate, lambd=lambd)

                hist = []
                for ep in range(EPOCHS):
                    hist = nn.train(exmple, expct)

                df = df.append(
                    {
                        'l_rate': l_rate,
                        'lambda': lambd,
                        'final_loss': hist[-1]
                    },
                    ignore_index=True
                )
    avg = df.groupby(['l_rate', 'lambda']).mean()
    print(avg.loc[avg['final_loss'].idxmin()])
    return avg


class Network:

    def __init__(self, num_input, num_hidden, num_output, l_rate=L_RATE, lambd=LAMBDA):
        self.weights1_2 = np.random.randn(num_hidden, num_input + 1)  # rand vs randn
        self.weights2_3 = np.random.randn(num_output, num_hidden + 1)
        self.biases = np.ones(2)
        self.a2 = None
        self.a3 = None
        self.z2 = None
        self.z3 = None
        self.input = None
        self.l_rate = l_rate
        self.lambd = lambd
        self.zero_grad()
        self.loss_history = []

    def zero_grad(self):
        self.d_weights1_2 = np.zeros(self.weights1_2.shape)
        self.d_weights2_3 = np.zeros(self.weights2_3.shape)

    def forward(self, x):
        self.input = x
        self.input = np.insert(self.input, 0, self.biases[0], axis=0)
        self.z2 = np.dot(self.weights1_2, self.input)
        self.a2 = sigmoid(self.z2)
        self.a2 = np.insert(self.a2, 0, self.biases[1], axis=0)
        self.z3 = np.dot(self.weights2_3, self.a2)
        self.a3 = sigmoid(self.z3)

        return self.a3

    def backward(self, expected):
        e3 = -(expected - self.a3) * d_sigmoid(self.a3)
        e2 = np.dot(self.weights2_3.T, e3) * d_sigmoid(self.a2)
        self.d_weights2_3 += e3.reshape(len(e3), 1) * self.a2
        self.d_weights1_2 += e2[1:].reshape(len(e2) - 1, 1) * self.input

    def step(self, batch_size):
        # weights
        self.weights2_3[:, 1:] = self.weights2_3[:, 1:] - self.l_rate * ((1. / batch_size) * self.d_weights2_3[:, 1:] + self.lambd * self.weights2_3[:, 1:])
        self.weights1_2[:, 1:] = self.weights1_2[:, 1:] - self.l_rate * ((1. / batch_size) * self.d_weights1_2[:, 1:] + self.lambd * self.weights1_2[:, 1:])

        # biases
        self.weights2_3[:, 0] = self.weights2_3[:, 0] - self.l_rate * ((1. / batch_size) * self.d_weights2_3[:, 0])
        self.weights1_2[:, 0] = self.weights1_2[:, 0] - self.l_rate * ((1. / batch_size) * self.d_weights1_2[:, 0])

        self.zero_grad()

    def train(self, examples, expected):
        run_loss = 0.0
        for i in range(len(examples)):
            out = self.forward(examples[:, i])
            run_loss += loss(out, expected[:, i])
            self.backward(expected[:, i])

        self.loss_history.append(run_loss/len(examples))
        self.step(batch_size=len(examples))

        return self.loss_history


if __name__ == '__main__':
    # Hyper Parameter Optimization
    # hyper_parameter_optimization(runs=10)

    nn = Network(NUM_INPUTS, NUM_HID, NUM_INPUTS)
    example = expect = np.eye(NUM_INPUTS)

    history = None
    for ep in range(EPOCHS):
        history = nn.train(example, expect)
        if ep % 200 == 0:
            print(f"epoch: {ep}\tloss: {history[-1]}")

    hidden_activations = []
    for i in range(len(example)):
        out = nn.forward(example[:, i])
        hidden_activations.append(nn.a2[1:])
        # test model
        '''
        print('expected ', example[:, i])
        print('out ', out)
        print('mse ', loss(out, example[:, i]))
        '''
        
    # plot activations
    '''
    print(hidden_activations)
    plot_activations(hidden_activations)
    '''

    # plot weights
    '''
    print(nn.weights1_2)
    print(nn.weights2_3)
    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(nn.weights1_2.T)
    axes[1].matshow(nn.weights2_3.T)
    plt.show()
    '''

    # plot loss
    '''
    plt.plot(history, label=f"lr: {L_RATE}, ld: {LAMBDA}", linewidth=2)
    plt.legend()
    plt.show()
    '''