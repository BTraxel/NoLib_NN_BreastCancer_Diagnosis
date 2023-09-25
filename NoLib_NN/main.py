from config import Config
from neural_network import NeuralNetwork
from data import Data

def main():
    config = Config()
    config.ask_parameters()
    data = Data()
    data.get_data()
    data.split_train_test()
    nn = NeuralNetwork(config, data)
    nn.simpleBenchmark()
    nn.benchmark()

if __name__ == "__main__":
    main()