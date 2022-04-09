import numpy as np


def softmax_crossentropy_with_logits(logits, labels):
    logits_for_answers = logits[np.arange(len(logits)), labels]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
    return xentropy


def grad_softmax_crossentropy_with_logits(logits, labels):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), labels] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    return (- ones_for_answers + softmax) / logits.shape[0]


def forward(network, X):
    # wylicza wartości funkcji aktywacji neuronu
    activations = []
    input = X

    for layer in network:
        activations.append(layer.forward(input))
        # aktualizujemy listę wejściową
        input = activations[-1]

    assert len(activations) == len(network)
    return activations


def predict(network, X):
    # zwraca najbardziej prawdopodobny dopasowany element
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y):
    # Trenujemy siec "network" zbiorem "X" z kluczami "y"

    # aktywujemy warstwy
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations
    logits = layer_activations[-1]

    # wyliczamy funkcje straty i wstepny gradient
    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    # możemy wywołac layer.backward która przechodzi z ostatniej do pierwszej warstwy
    # wykonujemy wsteczną propagację
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]

        loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)

    return np.mean(loss)
