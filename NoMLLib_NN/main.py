from config import Config
from neural_network import NeuralNetwork
from data import Data
import numpy as np

def main():
    #Create a config object
    config = Config()
    #Set the config object using user input
    config.ask_parameters()
    #Create a data object
    data = Data()
    #Set the data object by extracting the data from a csv file
    data.get_data()
    #Split the data into training and testing set
    data.split_train_test()
    #Create a neural network using the config object to set the hyperparameters
    nn = NeuralNetwork(config)
    #Train and test the parametred neural network on the data fed
    nn.simpleBenchmark(data)
    #Benchmark can be use to drow plot helping you visualise the neural network performance and find the best hyperparameters for your case
    nn.benchmark(1, data)
    nn.benchmark(2, data)
    nn.benchmark(3, data)
    nn.benchmark(4, data)
    nn.benchmark(5, data)
    #Train the neural network using desired data
    nn.train(data)
    #Predict the result of a set of data, the neural network must have beem trained and the data must have the same number of feature as the data used in training
    X_Predict_Test = np.array([data.Xtr[0]])
    print(nn.predict(X_Predict_Test))


if __name__ == "__main__":
    main()