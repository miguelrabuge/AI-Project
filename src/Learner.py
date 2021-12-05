import os
import loader
import numpy as np
import matplotlib.pyplot as plt
from AlexNet import AlexNet_cnn
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Learner():
    def __init__(self, X_pool, y_pool, X_val, y_val, n_initial, estimator: KerasClassifier, epochs=5, strategy=uncertainty_sampling):
        def get_initial_data(X, y, n):
            # Randomly choose initial training indexes
            initial_idxs = np.random.choice(range(len(X)), size=n, replace=False)
            X_init = X[initial_idxs]
            y_init = y[initial_idxs]
            # Remove Queried Instances from the X,y Pools
            X = np.delete(X, initial_idxs, axis=0)
            y = np.delete(y, initial_idxs, axis=0)
            return X, y, X_init, y_init

        self.estimator = estimator
        self.strategy = strategy
        self.X_pool, self.y_pool, X_init, y_init = get_initial_data(X_pool, y_pool, n_initial)
        self.X_val, self.y_val = X_val, y_val
        self.learner = ActiveLearner(estimator=estimator, query_strategy=strategy, X_training=X_init, y_training=y_init, epochs=epochs)

    def learn(self, n_queries, n_instances, epochs=5):
        trains, tests = [], []
        try:
            for i in range(n_queries):
                # Query instances
                idxs, instances = self.learner.query(self.X_pool, n_instances=n_instances, verbose=0)
                x = self.X_pool[idxs]
                y = self.y_pool[idxs]

                print(f"Query Nº {i}", end="\n")

                # Teach Model the query instances
                self.learner.teach(X=x, y=y, only_new=False, epochs=epochs)

                # Add Train and Test Scores
                trains.append(round(self.learner.score(self.learner.X_training, self.learner.y_training), 3))
                tests.append(round(self.learner.score(self.X_val, self.y_val), 3))
                
                # Remove Queried Instances from the X,y Pools
                self.X_pool = np.delete(self.X_pool, idxs, axis=0)
                self.y_pool = np.delete(self.y_pool, idxs, axis=0)
        except:
            print(f"Active Learning Loop Ended: Not enough datapoints (has {self.X_pool.shape[0]} needs {n_instances}) for another query loop")
        return trains, tests

    def predict(self, X):
        return self.learner.predict(X)

    def score(self, X, y):
        return self.learner.score(X, y)

def main(estimator, epochs, n_init=600, n_queries=11, query_size=60):
    # Getting Data
    X_train, y_train, X_val, y_val = loader.main()
    
    # Defining Active Learner Model
    learner = Learner(X_train, y_train, X_val, y_val, n_init, estimator=estimator, epochs=epochs)
    return learner.learn(n_queries, query_size, epochs=epochs)

if __name__ == "__main__":
    np.random.seed(42)
    main(AlexNet_cnn(), 25)