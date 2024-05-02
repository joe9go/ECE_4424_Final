from math import exp
import random
random.seed(1)

def logistic_func(x):
    s = (1+exp(-x))**-1
    return s

def dot(x, y):
    s = sum((a * b) for (a,b) in zip(x,y))
    return s

def predict(model, point):
    p = logistic_func(dot(model, point))
    return p

def initialize_model(k):
    return [0.5 for x in range(k)]
    
class class_logistic:

    model = None

    def __init__(self):
        pass

    
    def predict(self, features):
        out = []
        for point in features:
            out.append(predict(self.model, point))

        return out
            
    def fit(self, train_features, train_labels):

        epochs = 50
        rate = 2e-3
        lam = 3e-3
        
        num_features = len(train_features[0])
        model = initialize_model(num_features)
        for e in range(epochs):
          learn_rate = rate * ((epochs-e)/epochs)
          for n in range(len(train_features)):
            new_weights = [0]*num_features

            index = random.randint(0,len(train_features)-1)
            point = train_features[index]
            Y = train_labels[index]
            P = predict(model, point)
            for i in range(num_features):
                gradient = point[i] * (Y-P)
                new_weights[i] = model[i] - learn_rate*lam*model[i] + learn_rate*gradient

            model = new_weights

        self.model = model
        


        
