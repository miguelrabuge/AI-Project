from tensorflow import random
import numpy as np
import argparse
import AlexNet 
import flying_dogs 
import LeNet_5 
import Learner

def seed_loop(func, epochs, n_seeds=30):
    trains, tests = [], []
    for seed in range(n_seeds):
        print(f"Seed: {seed}")
        random.set_seed(seed)
        np.random.seed(seed)
        acc_train, acc_test = func(epochs)
        trains.append(round(acc_train, 3))
        tests.append(round(acc_test, 3))
    return sum(trains) / len(trains), sum(tests) / len(tests)

def al_seed_loop(estimator, epochs, n_seeds=30):
    trains, tests = [], []
    for seed in range(n_seeds):
        print(f"Seed: {seed}")
        random.set_seed(seed)
        acc_train, acc_test = Learner.main(estimator(), epochs)
        trains.append(acc_train)
        tests.append(acc_test)
    return trains, tests


if __name__ == "__main__":
    # Static Parameters
    EPOCHS = 25
    FILE = "run_results.txt"
    cl_mean_f = lambda l: [round(sum(sublist) / len(sublist), 3) for sublist in zip(*l)]

    # Argument Parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-A", help="Active Learning Loop", action="store_const", const=True)
    parser.add_argument("-a", help="AlexNet", action="store_const", const=[AlexNet.main, AlexNet.AlexNet_cnn])
    parser.add_argument("-f", help="Flying Dogs Net", action="store_const", const=[flying_dogs.main, flying_dogs.flying_dogs_cnn])
    parser.add_argument("-l", help="LeNet-5", action="store_const", const=[LeNet_5.main, LeNet_5.LeNet5_cnn])

    params = vars(parser.parse_args())

    active_learning = True if params.pop("A") else False
    with open(FILE, "w") as file:
        # Running CNNs
        for flag, func in params.items():
            if func:
                train_avg, test_avg = seed_loop(func[0], EPOCHS)
                print(f"[{func[0].__doc__}]\nTrain AVG: {train_avg}\nTest AVG: {test_avg}", file=file)

        # Running Active CNNs
        if active_learning:
            for flag, func in params.items():
                if func:
                    print(f"[{func[0].__doc__}] - Active Learning", file=file)
                    trains, tests = al_seed_loop(func[1], EPOCHS)
                    print(f"Train Matrix (N_seeds x N_queries)", file=file)
                    
                    for run in trains:
                        print(run, file=file)
                    print(f"Queries Average", file=file)
                    print(f"{cl_mean_f(trains)}", file=file)


                    print(f"Test Matrix (N_seeds x N_queries)", file=file)
                    for run in tests:
                        print(run, file=file)
                    print(f"Query Average", file=file)
                    print(f"{cl_mean_f(tests)}", file=file)



