def sgd(lr=1e-3):
    def _call(state, grads):
        return state - lr * grads

    return _call
