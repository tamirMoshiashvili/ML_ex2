import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    """
    :param x: vector.
    :return: vector after applying softmax function.
    """
    x -= np.max(x)  # for numeric stability
    x = np.exp(x)
    x /= np.sum(x)
    return x


class LogisticRegressionClassifier(object):
    def __init__(self, in_dim, labels):
        """
        :param in_dim: input dimension.
        :param labels: array-like, each is an optional label.
        """
        out_dim = len(labels)
        self.W = np.zeros((in_dim, out_dim))
        self.b = np.zeros(out_dim)

        self.l2i = {l: i for i, l in enumerate(labels)}

    def forward(self, x):
        """
        :param x: vector.
        :return: output of logistic regression.
        """
        return softmax(np.dot(x, self.W) + self.b)

    def predict_on(self, x):
        """
        :param x: input to classify.
        :return: scalar, predicted label of x.
        """
        return np.argmax(self.forward(x))

    def loss_and_gradients(self, x, y):
        """
        Compute the loss and the gradients at point x with given parameters.
        :param x: input to the classifier.
        :param y: scalar, indicating the correct label.
        :return: loss, [gW, gb] where:
                 - loss - scalar, loss of the classifier according to the prediction and the gold label.
                 - gW   - matrix, gradient of W.
                 - gb   - vector, gradient of b.
        """
        # loss
        x = np.array([x])
        probs = self.forward(x)
        gold_label_index = self.l2i[y]
        loss = -np.log(probs[gold_label_index])

        # gradient of b
        gb = np.copy(probs)
        gb[gold_label_index] -= 1

        # gradient of W
        gW = np.zeros(self.W.shape)
        for (i, j) in np.ndindex(self.W.shape):
            gW[i, j] = x[i] * probs[j] - (j == gold_label_index) * x[i]

        return loss, [gW, gb]

    def update_params(self, grads, lr=0.1):
        """
        :param grads: list of gradients of format [gW, gb].
        :param lr: learning rate.
        """
        self.W -= lr * grads[0]
        self.b -= lr * grads[1]


def produce_train_set(labels, size=100):
    """
    :param labels: list of integers, each is a label.
    :param size: size of samples from each label.
    :return: train set, list of tuples, each is (sample, label), of size |labels|*size .
    """
    train_set = []
    y_train = []

    for label in labels:
        loc = 2 * label
        xs = np.random.normal(loc, size=size)

        train_set.extend(xs)
        y_train.extend([label] * size)

    return zip(train_set, y_train)


def train_classifier(cls, labels, epochs=10):
    """
    Update the classifiers parameters according to a train-set.
    :param cls: classifier object.
    :param labels: list of integers, each is label.
    :param epochs: num epochs.
    """
    # create train set
    train_set = produce_train_set(labels)

    for _ in range(epochs):
        np.random.shuffle(train_set)
        for x, y in train_set:
            _, grads = cls.loss_and_gradients(x, y)
            cls.update_params(grads)


def normal_of(x, loc, scale=1.0):
    """
    :param x: input.
    :param loc: centre.
    :param scale: scale.
    :return: probability.
    """
    numerator = np.exp((-(x - loc) ** 2) / (2 * scale ** 2))
    denominator = np.sqrt(2 * np.pi * (scale ** 2))
    return numerator / denominator


def cmp_dists(cls, labels, show_results=False):
    inspace = np.linspace(0, 10, 300)
    probs_gold = []
    probs_pred = []

    # calculate probability of gold and pred
    for x in inspace:
        normals = [normal_of(x, 2 * label) for label in labels]
        probs_gold.append(normals[0] / sum(normals))
        probs_pred.append(cls.forward(x)[0][0])

    # plot results
    if show_results:
        plt.xlabel('x')
        plt.ylabel('P(x | y=1)')
        plt.plot(inspace, probs_gold, 'b', inspace, probs_pred, 'r')
        plt.legend(('gold', 'pred'))
        plt.show()


def main():
    labels = [1, 2, 3]
    in_dim = 1
    cls = LogisticRegressionClassifier(in_dim, labels)

    train_classifier(cls, labels)
    cmp_dists(cls, labels, show_results=True)


if __name__ == '__main__':
    main()
