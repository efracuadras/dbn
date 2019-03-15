import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification
from Rafd import Rafd


# Splitting data
rafd = Rafd("entrenamiento/")
X_train, X_test, Y_train, Y_test = rafd.getData()

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.001,
                                         n_epochs_rbm=15,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='sigmoid',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))

