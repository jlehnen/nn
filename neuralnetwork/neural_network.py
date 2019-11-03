import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(linewidth=400)

L_RATE = 0.5
LAMBDA = 0.00001
EPOCHS = 5000
NUM_INPUTS = 8
NUM_HID = 3


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1. - x)


def loss(h, y):
    return (np.square(h - y)).mean()


def hyper_parameter_optimization(verbose=True):
    # np.random.seed(42690)
    exmple = expct = np.eye(NUM_INPUTS)
    results = []
    for l_rate in tqdm([0.1, 0.3, 0.5, 0.7], disable=not verbose):
        for lambd in [0.00001, 0.0001, 0.005]:
            nn = Network(NUM_INPUTS, NUM_HID, NUM_INPUTS)

            hist = []
            for ep in range(EPOCHS):
                hist = nn.train(exmple, expct)

            results.append(
                {
                    'l_rate': l_rate,
                    'lambda': lambd,
                    'loss': hist
                }
            )

    best_config = results[0]
    for result in results:
        if verbose:
            print(f"l_rate: {result['l_rate']}, lambda: {result['lambda']}, loss: {result['loss'][-1]}")
        if result['loss'][-1] < best_config['loss'][-1]:
            best_config = result
        plt.plot(result['loss'], label=f"lr: {result['l_rate']}, ld: {result['lambda']}", linewidth=2)
    plt.legend()
    if verbose:
        print(f"best config: l_rate: {best_config['l_rate']}, lambda: {best_config['lambda']}, loss: {best_config['loss'][-1]}")
        plt.show()
    return best_config


class Network:

    def __init__(self, num_input, num_hidden, num_output):
        self.weights1_2 = np.random.rand(num_hidden, num_input + 1)  # *1./50
        self.weights2_3 = np.random.rand(num_output, num_hidden + 1)  # *1./50
        self.biases = np.ones(2)
        self.a2 = None
        self.a3 = None
        self.z2 = None
        self.z3 = None
        self.input = None
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

    def backward(self, out, _expected):
        e3 = -(_expected - out) * d_sigmoid(self.a3)
        e2 = np.dot(self.weights2_3.T, e3) * d_sigmoid(self.a2)
        self.d_weights2_3 += e3.reshape(len(e3), 1) * self.a2
        self.d_weights1_2 += e2[1:].reshape(len(e2) - 1, 1) * self.input

    def step(self, batch_size):
        # weights
        self.weights2_3[:, 1:] = self.weights2_3[:, 1:] - L_RATE * ((1. / batch_size) * self.d_weights2_3[:, 1:] + LAMBDA * self.weights2_3[:, 1:])
        self.weights1_2[:, 1:] = self.weights1_2[:, 1:] - L_RATE * ((1. / batch_size) * self.d_weights1_2[:, 1:] + LAMBDA * self.weights1_2[:, 1:])

        # biases
        self.weights2_3[:, 0] = self.weights2_3[:, 0] - L_RATE * ((1. / batch_size) * self.d_weights2_3[:, 0])
        self.weights1_2[:, 0] = self.weights1_2[:, 0] - L_RATE * ((1. / batch_size) * self.d_weights1_2[:, 0])

        self.zero_grad()

    def train(self, examples, expected):
        run_loss = 0.0
        for i in range(len(examples)):
            out = self.forward(examples[:, i])
            run_loss += loss(out, expected[:, i])
            self.backward(out, expected[:, i])

        self.loss_history.append(run_loss/len(examples))
        self.step(batch_size=len(examples))

        return self.loss_history


if __name__ == '__main__':
    '''
    nn = Network(NUM_INPUTS, NUM_HID, NUM_INPUTS)
    example = expect = np.eye(NUM_INPUTS)

    for ep in range(EPOCHS):
        history = nn.train(example, expect)
        if ep % 200 == 0:
            print(f"epoch: {ep}\tloss: {history[-1]}")

    print(history)
    for i in range(len(example)):
        out = nn.forward(example[:, i])
        print('expected ', example[:, i])
        print('out ', out)
        print('mse ', loss(out, example[:, i]))
    '''
    hyper_parameter_optimization()
    '''
    best_configs = []
    for i in tqdm(range(10)):
        best_configs.append(hyper_parameter_optimization(verbose=False))
    for conf in best_configs:
        print(f"best config: l_rate: {conf['l_rate']}, lambda: {conf['lambda']}, loss: {conf['loss'][-1]}")
    '''