from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score

import keras

import numpy as np

import matplotlib.pyplot as plt

print("loading data")
data = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(data['data'], data['target'], test_size=.2)

#results format [[[accuracy, precision, RMSE, mAE, F-score] for each test run] for each model]
results = []

def evaluate_model(preds, labels):

    out = []
    
    out.append(accuracy_score(labels, preds))
    out.append(precision_score(labels, preds, average="micro"))
    out.append(root_mean_squared_error(labels, preds))
    out.append(mean_absolute_error(labels, preds))
    out.append(f1_score(labels, preds, average="micro"))

    return out
        

#xgboost, not so much an idea for a model as the name of a library, so we use that
from xgboost import XGBClassifier
def test_xgboost(train_features, train_labels, validation_features, validation_labels):

    model = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    model.fit(train_features, train_labels)

    preds = model.predict(validation_features)

    result = evaluate_model(preds, validation_labels)

    return result

from sklearn.svm import SVC as support_vector_model
def test_support_vector(train_features, train_labels, validation_features, validation_labels):
    
    model = make_pipeline(StandardScaler(), support_vector_model(gamma='auto'))
    model.fit(train_features, train_labels)

    preds = model.predict(validation_features)

    result = evaluate_model(preds, validation_labels)
    
    return result

from sklearn.ensemble import RandomForestClassifier
def test_random_forest(train_features, train_labels, validation_features, validation_labels):

    model = RandomForestClassifier()
    model.fit(train_features, train_labels)

    preds = model.predict(validation_features)

    result = evaluate_model(preds, validation_labels)
    
    return result

#"neural network" is vague, i'll try 2 dense layers
def test_neural_network(train_features, in_train_labels, validation_features, in_validation_labels):

    train_labels = [[]]*len(in_train_labels)
    for i in range(len(in_train_labels)):
        train_labels[i] = [0]*4
        train_labels[i][in_train_labels[i]] = 1
        
    train_labels = np.array(train_labels)
    
    validation_labels = [[]]*len(in_validation_labels)
    for i in range(len(in_validation_labels)):
        validation_labels[i] = [0]*4
        validation_labels[i][in_validation_labels[i]] = 1

    validation_labels = np.array(validation_labels)
    
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=train_features.shape[1:]))
    
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    
    model.add(keras.layers.Dense(4))
    model.add(keras.layers.Activation('softmax'))
      
    #Adam optimizer
    opt = keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    print(train_features.shape)

    model.fit(train_features, train_labels,
             batch_size = 1024,
             epochs = 50,
             validation_data=(validation_features, validation_labels),
             shuffle = True)

    res = model.predict(validation_features)

    preds = []
    for i in res:
        m = list(i)
        preds.append([0]*4)
        preds[-1][m.index(max(m))]=1

    result = evaluate_model(preds, validation_labels)

    return result

#like homework 2
from sklearn.linear_model import LogisticRegression
def test_logistic_regression(train_features, train_labels, validation_features, validation_labels):

    result = []

    model = LogisticRegression()
    model.fit(train_features, train_labels)

    preds = model.predict(validation_features)

    result = evaluate_model(preds, validation_labels)
    
    return result

from class_logistic import class_logistic
def test_class_logistic(train_features, train_labels, validation_features, validation_labels):

    result = []

    model = class_logistic()
    model.fit(train_features, train_labels)

    preds = model.predict(validation_features)

    result = evaluate_model(preds, validation_labels)
    
    return result


print("running xgboost")
results.append(test_xgboost(X_train, Y_train, X_test, Y_test))


print("running support vector")
results.append(test_support_vector(X_train, Y_train, X_test, Y_test))


print("running random forest")
results.append(test_random_forest(X_train, Y_train, X_test, Y_test))


print("running neural network")
results.append(test_neural_network(X_train, Y_train, X_test, Y_test))


print("running logistic")
results.append(test_logistic_regression(X_train, Y_train, X_test, Y_test))

arrRes = np.array(results)

#results format [[[accuracy, precision, RMSE, mAE, F-score] for each test run] for each model]
barRes = {
    'xgboost': arrRes[0],
    'support vector': arrRes[1],
    'random forest': arrRes[2],
    'neural network': arrRes[3],
    'logistic': arrRes[4]
}

fig, ax = plt.subplots(layout='constrained')

metrics = ['Accuracy', 'Precision', 'RMSE', 'MAE', 'F-score']

x = np.arange(len(metrics))
barWidth = .1
n = 0

for val, score in barRes.items():
    rects = ax.bar(x+barWidth*n, score, barWidth, label=val)
    n+=1
    
ax.set_xticks(x + barWidth, metrics)

ax.legend(loc='upper center')

plt.show(cmap="Greys")
