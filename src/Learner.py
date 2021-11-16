import os
import loader
import numpy as np
import matplotlib.pyplot as plt
from AlexNet import AlexNet_cnn
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)

class Learner():
    def __init__(self, X_pool, y_pool, X_val, y_val, n_initial, estimator: KerasClassifier, strategy=uncertainty_sampling):
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
        self.learner = ActiveLearner(estimator=estimator, query_strategy=strategy, X_training=X_init, y_training=y_init)

    def learn(self, n_queries, n_instances, epochs=5, plot=False):
        scores = []
        try:
            for i in range(n_queries):
                # Query instances
                idxs, instances = self.learner.query(self.X_pool, n_instances=n_instances, verbose=0)
                x = self.X_pool[idxs]
                y = self.y_pool[idxs]

                print(f"Query Nº {i}", end="\n")

                # Teach Model the query instances
                self.learner.teach(X=x, y=y, only_new=False, epochs=epochs)

                scores.append(self.learner.score(self.X_val, self.y_val))
                # Remove Queried Instances from the X,y Pools
                self.X_pool = np.delete(self.X_pool, idxs, axis=0)
                self.y_pool = np.delete(self.y_pool, idxs, axis=0)
        except:
            print(f"Active Learning Loop Ended: Not enough datapoints (has {self.X_pool.shape[0]} needs {n_instances}) for another query loop")
        if plot:
            plt.plot(range(len(scores)), scores, )
            plt.title("Scores")
            plt.xlabel("Query Number")
            plt.ylabel("Accuracy")
            plt.legend(["Validation"])
            plt.show()
        return self

    def predict(self, X):
        return self.learner.predict(X)

    def score(self, X, y):
        return self.learner.score(X, y)

if __name__ == "__main__":
    # Getting Data
    X_train, y_train, X_val, y_val = loader.main()
    
    # Defining Active Learner Model
    learner = Learner(X_train, y_train, X_val, y_val, 600, estimator=AlexNet_cnn())
    learner.learn(11, 60, epochs=10, plot=True)

    print(f"Final Validation Score: {learner.score(X_val, y_val)}")