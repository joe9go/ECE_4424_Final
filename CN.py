class chaotic_neuronal:

    e = 0.05

    def __init__(self):
        pass

    def fit(self, train_features, train_labels):
        train_features = norm(train_features)

    def predict(self, features):
        return [0]*(len(features))

    def TT(self, X):
        pass

def GLS_neuron(x, b = 0.467354):
    
    if (x < b):
        return x/b
    else:
        return (1-x)/(1-b)


def norm(X):

    mn = min(X)
    mx = max(X)

    return (X - mn)/(mx-mn)
    
