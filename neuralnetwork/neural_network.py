import numpy as np

np.set_printoptions(linewidth=400)

L_RATE = 0.5
LAMBDA = 0.00001
EPOCHS = 10000


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1. - x)


def loss(h, y):
    return (np.square(h - y)).mean()


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
        # TODO cleaner bias handling?
        self.input = np.insert(self.input, 0, self.biases[0], axis=0)
        #print('w1_2', self.weights1_2)
        #print('input', self.input)
        self.z2 = np.dot(self.weights1_2, self.input)
        #print('z2', self.z2)
        self.a2 = sigmoid(self.z2)
        #print('a2b', self.a2)
        self.a2 = np.insert(self.a2, 0, self.biases[1], axis=0)
        #print('a2', self.a2)

        self.z3 = np.dot(self.weights2_3, self.a2)
        #print('dot(w2_3, a2)', self.z3)
        self.a3 = sigmoid(self.z3)

        return self.a3

    def backward(self, out, _expected):
        #print('out', out)
        #print('exp', _expected)
        e3 = -(_expected - out) * d_sigmoid(self.a3)
        #print('sig z3', d_sigmoid(self.z3))
        #print('e3', e3)
        #print('w2_3', self.weights2_3.T)
        #print('dot(w2_3, e3)', np.dot(self.weights2_3.T, e3))
        #print('a2', self.a2)
        #print('a2', self.a2)
        #print('d_sig(a2)', d_sigmoid(self.a2))
        #print('insert', np.insert(d_sigmoid(self.z2), 0, self.biases[1], axis=0))
        e2 = np.dot(self.weights2_3.T, e3) * d_sigmoid(self.a2)
        #print('dot(w2_3, e3) * thing', e2)
        # e2 = np.dot(self.weights2_3[:, 1:].T, e3) * d_sigmoid(self.z2)
        self.d_weights2_3 += e3.reshape(len(e3), 1) * self.a2
        #print('e2', e2)
        #print('post', e2[1:].reshape(len(e2) - 1, 1))
        #print('input', self.input)
        #print('update',  e2[1:].reshape(len(e2) - 1, 1) * self.input)
        self.d_weights1_2 += e2[1:].reshape(len(e2) - 1, 1) * self.input
        #print('dw', self.d_weights1_2)

    def step(self, batch_size):
        # weights
        self.weights2_3[:, 1:] = self.weights2_3[:, 1:] - L_RATE * ((1. / batch_size) * self.d_weights2_3[:, 1:] + LAMBDA * self.weights2_3[:, 1:])
        self.weights1_2[:, 1:] = self.weights1_2[:, 1:] - L_RATE * ((1. / batch_size) * self.d_weights1_2[:, 1:] + LAMBDA * self.weights1_2[:, 1:])

        # biases
        self.weights2_3[:, 0] = self.weights2_3[:, 0] - L_RATE * ((1. / batch_size) * self.d_weights2_3[:, 0])
        self.weights1_2[:, 0] = self.weights1_2[:, 0] - L_RATE * ((1. / batch_size) * self.d_weights1_2[:, 0])

        self.zero_grad()
        #print('zero', self.d_weights1_2)

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
    num_inputs = 8
    num_hid = 3
    nn = Network(num_inputs, num_hid, num_inputs)
    example = expect = np.eye(num_inputs)

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