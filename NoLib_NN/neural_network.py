import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        self.trained = False

    def set_random_weights(self):
        # Initialize neural network weights
        # Input layer weights
        self.Weights_Input = np.random.uniform(-1, 1, (self.data.X.shape[1], self.config.num_hidden_neurons_layer))* np.sqrt(2 / self.data.X.shape[1])
        self.Update_Input_Batch = np.zeros((self.data.X.shape[1], self.config.num_hidden_neurons_layer))
        # Hidden layer weights
        self.Weights_Hiden = np.random.uniform(-1, 1, self.config.num_hidden_neurons_layer)* np.sqrt(2 / self.config.num_hidden_neurons_layer)
        self.Update_Hiden_Batch = np.zeros(self.config.num_hidden_neurons_layer)
        # Hidden layer weights for multiple layers
        self.WeightsHidenp = np.empty(self.config.num_hidden_layers - 1)
        self.WeightsHidenp = np.array(self.WeightsHidenp, dtype=object)
        self.Update_Hiden_Batchp = np.empty(self.config.num_hidden_layers - 1)
        self.Update_Hiden_Batchp = np.array(self.Update_Hiden_Batchp, dtype=object)

        for i in range(self.config.num_hidden_layers-1):
            self.WeightsHidenp[i] = np.random.uniform(-1, 1, (self.config.num_hidden_neurons_layer, self.config.num_hidden_neurons_layer))* np.sqrt(2 / self.config.num_hidden_neurons_layer)
            self.Update_Hiden_Batchp[i] = np.zeros((self.config.num_hidden_neurons_layer, self.config.num_hidden_neurons_layer))

        self.Update_Hiden_BatchpCopy = self.Update_Hiden_Batchp.copy()

    def train(self, data):
        # Train the neural network
        self.data = data
        self.reset()
        if (self.config.Type_gradient_descent == 1):  # Stochastic
            self.stochastic_gradient_descent()
        
        elif (self.config.Type_gradient_descent == 2):  # Batch
            self.batch_gradient_descent()
        
        elif (self.config.Type_gradient_descent == 3):  # Mini-batch
            self.mini_batch_gradient_descent()
        self.trained = True

    
    def test(self, data):
        # Test the neural network
        self.start_time = time.time()
        self.train(data)
        self.TrainingTime = time.time() - self.start_time
        self.verify()

    def simpleBenchmark(self, data):
        # Run a simple benchmark and print results
        self.test(data)
        print("Training time (Log Reg using Gradient descent):" + str(self.TrainingTime) + " seconds")
        print("Learning rate: {}\nIteration: {}".format(self.config.Learning_rate, self.config.num_epochs_training))

        self.result = pd.DataFrame({'Result': self.FE2})
        self.objective = pd.DataFrame({'Objective': self.data.yt})
        self.f = pd.DataFrame(np.around(self.result, decimals=6)).join(self.objective)
        self.f['pred'] = self.f['Result'].apply(lambda x: 0 if x < 0.5 else 1)
        print(self.f)
        print("Accuracy (Loss minimization):")
        print(self.f.loc[self.f['pred'] == self.f['Objective']].shape[0] / self.f.shape[0] * 100)

        self.YActual = self.f['Objective'].tolist()
        self.YPredicted = self.f['pred'].tolist()
        self.TraceConfusionMatrix()

    def predict(self, X):
        # Function to predict a tumor nature
        if(self.trained == True):
            results = []  # Store the results for all samples

            for sample in range(X.shape[0]):
                for node in range(self.config.num_hidden_neurons_layer):
                    self.preActiv_Hp[0][node] = np.dot(X[sample], self.Weights_Input[:, node])
                    self.postActiv_Hp[0][node] = self.activationfunction(self.preActiv_Hp[0][node], self.config.Type_activation_function)

                for layer in range(1, self.config.num_hidden_layers):
                    for node in range(self.config.num_hidden_neurons_layer):
                        self.preActiv_Hp[layer][node] = np.dot(self.postActiv_Hp[layer - 1], self.WeightsHidenp[layer - 1][:, node])
                        self.postActiv_Hp[layer][node] = self.activationfunction(self.preActiv_Hp[layer][node], self.config.Type_activation_function)

                preActivation_O = np.dot(self.postActiv_Hp[-1], self.Weights_Hiden)
                postActivation_O = self.activationfunction(preActivation_O, self.config.Type_activation_function)

                results.append(postActivation_O)

            return results
        else:
            return("Neural Network not trained")

    def reset(self):
        self.set_random_weights()
        self.reset_array()
    
    def stochastic_gradient_descent(self):
        for epoch in range(self.config.num_epochs_training):
            for sample in range(self.data.SizeofTrain):
                # Loop for multiple hiden layer
                for node in range(self.config.num_hidden_neurons_layer):
                    self.preActiv_Hp[0][node] = np.dot(self.data.Xtr[sample, :], self.Weights_Input[:, node])
                    self.postActiv_Hp[0][node] = self.activationfunction(self.preActiv_Hp[0][node], self.config.Type_activation_function)

                for layer in range(1, self.config.num_hidden_layers):
                    for node in range(self.config.num_hidden_neurons_layer):
                        self.preActiv_Hp[layer][node] = np.dot(self.postActiv_Hp[layer - 1], self.WeightsHidenp[layer - 1][:, node])
                        self.postActiv_Hp[layer][node] = self.activationfunction(self.preActiv_Hp[layer][node],self.config.Type_activation_function)

                preActivation_O = np.dot(self.postActiv_Hp[-1], self.Weights_Hiden)
                postActivation_O = self.activationfunction(preActivation_O, self.config.Type_activation_function)

                FE = self.lossfunction(postActivation_O, self.data.ytr[sample], self.config.Type_loss_function)

                for H_node in range(self.config.num_hidden_neurons_layer):
                    self.s_error = FE * self.activationfunctionp(preActivation_O, self.config.Type_activation_function)
                    gradient_HtoD = self.s_error * self.postActiv_Hp[-1][H_node]

                    # Before last hidden
                    if (self.config.num_hidden_layers > 1):
                        # Before last hidden
                        for I_node in range(self.config.num_hidden_neurons_layer):
                            input_value = self.postActiv_Hp[-2][I_node]
                            gradient_Itod = self.s_error * self.Weights_Hiden[H_node] * self.activationfunctionp(
                                self.preActiv_Hp[-1][H_node], self.config.Type_activation_function) * input_value
                            self.s_errorp[-1][H_node] = self.activationfunctionp(self.preActiv_Hp[-1][H_node],self.config.Type_activation_function) * input_value
                            self.WeightsHidenp[-1][I_node, H_node] -= self.config.Learning_rate * gradient_Itod
                    else:
                        for I_node in range(self.data.X.shape[1]):
                            input_value = self.data.Xtr[sample, I_node]
                            gradient_Itod = self.s_error * self.Weights_Hiden[H_node] * self.activationfunctionp(self.preActiv_Hp[0][H_node],self.config.Type_activation_function) * input_value
                            self.Weights_Input[I_node, H_node] -= self.config.Learning_rate * gradient_Itod

                    self.Weights_Hiden[H_node] -= self.config.Learning_rate * gradient_HtoD

                if (self.config.num_hidden_layers > 1):
                    # Middle hiddens
                    for HidenLayer in range(self.config.num_hidden_layers - 3, -1, -1):
                        for H_node in range(self.config.num_hidden_neurons_layer):
                            for I_node in range(self.config.num_hidden_neurons_layer):
                                input_value = self.postActiv_Hp[HidenLayer][I_node]
                                self.s_errorp[HidenLayer][H_node] = self.activationfunctionp(self.preActiv_Hp[HidenLayer + 1][H_node],self.config.Type_activation_function) * input_value
                                gradient_Itod = np.dot(self.s_errorp[HidenLayer + 1],self.WeightsHidenp[HidenLayer + 1][H_node]) * self.s_errorp[HidenLayer][H_node]
                                self.WeightsHidenp[HidenLayer][I_node, H_node] -= self.config.Learning_rate * gradient_Itod

                        # First hidden
                    for H_node in range(self.config.num_hidden_neurons_layer):
                        for I_node in range(self.data.X.shape[1]):
                            input_value = self.data.Xtr[sample, I_node]
                            gradient_Itod = np.dot(self.s_errorp[0], self.WeightsHidenp[0][H_node]) * input_value
                            self.Weights_Input[I_node, H_node] -= self.config.Learning_rate * gradient_Itod

            if (self.config.Learning_rate_schedule == ("True")):
                self.config.Learning_rate = self.config.Learning_rate * (1.0 / (1.0 + self.config.Decay * epoch))
        return

    def batch_gradient_descent(self):
        for epoch in range(self.config.num_epochs_training):
            for sample in range(self.data.SizeofTrain):
                for node in range(self.config.num_hidden_neurons_layer):
                    self.preActiv_Hp[0][node] = np.dot(self.data.Xtr[sample, :], self.Weights_Input[:, node])
                    self.postActiv_Hp[0][node] = self.activationfunction(self.preActiv_Hp[0][node], self.config.Type_activation_function)

                for layer in range(1, self.config.num_hidden_layers):
                    for node in range(self.config.num_hidden_neurons_layer):
                        self.preActiv_Hp[layer][node] = np.dot(self.postActiv_Hp[layer - 1], self.WeightsHidenp[layer - 1][:, node])
                        self.postActiv_Hp[layer][node] = self.activationfunction(self.preActiv_Hp[layer][node],self.config.Type_activation_function)

                self.preActivation_O = np.dot(self.postActiv_Hp[-1], self.Weights_Hiden)
                self.postActivation_O = self.activationfunction(self.preActivation_O, self.config.Type_activation_function)

                self.FE = self.lossfunction(self.postActivation_O, self.data.ytr[sample], self.config.Type_loss_function)

                for H_node in range(self.config.num_hidden_neurons_layer):
                    self.s_error = self.FE * self.activationfunctionp(self.preActivation_O, self.config.Type_activation_function)
                    gradient_HtoD = self.s_error * self.postActiv_Hp[-1][H_node]

                    # Before last hidden
                    if (self.config.num_hidden_layers > 1):
                        # Before last hidden
                        for I_node in range(self.config.num_hidden_neurons_layer):
                            input_value = self.postActiv_Hp[-2][I_node]
                            gradient_Itod = self.s_error * self.Weights_Hiden[H_node] * self.activationfunctionp(
                                self.preActiv_Hp[-1][H_node], self.config.Type_activation_function) * input_value
                            self.s_errorp[-1][H_node] = self.activationfunctionp(self.preActiv_Hp[-1][H_node],self.config.Type_activation_function) * input_value
                            self.Update_Hiden_Batchp[-1][I_node, H_node] += gradient_Itod
                    else:
                        for I_node in range(self.data.X.shape[1]):
                            input_value = self.data.Xtr[sample, I_node]
                            gradient_Itod = self.s_error * self.Weights_Hiden[H_node] * self.activationfunctionp(self.preActiv_Hp[0][H_node],self.config.Type_activation_function) * input_value
                            self.Update_Input_Batch[I_node, H_node] += gradient_Itod

                    self.Update_Hiden_Batch[H_node] += gradient_HtoD

                if (self.config.num_hidden_layers > 1):
                    # Middle hiddens
                    for HidenLayer in range(self.config.num_hidden_layers - 3, -1, -1):
                        for H_node in range(self.config.num_hidden_neurons_layer):
                            for I_node in range(self.config.num_hidden_neurons_layer):
                                input_value = self.postActiv_Hp[HidenLayer][I_node]
                                self.s_errorp[HidenLayer][H_node] = self.activationfunctionp(self.preActiv_Hp[HidenLayer + 1][H_node],self.config.Type_activation_function) * input_value
                                self.gradient_Itod = np.dot(self.s_errorp[HidenLayer + 1],self.WeightsHidenp[HidenLayer + 1][H_node]) * self.s_errorp[HidenLayer][H_node]
                                self.Update_Hiden_Batchp[HidenLayer][I_node, H_node] += self.gradient_Itod

                        # First hidden
                    for H_node in range(self.config.num_hidden_neurons_layer):
                        for I_node in range(self.data.X.shape[1]):
                            self.input_value = self.data.Xtr[sample, I_node]
                            self.gradient_Itod = np.dot(self.s_errorp[0], self.WeightsHidenp[0][H_node]) * self.input_value
                            self.Update_Input_Batch[I_node, H_node] += self.gradient_Itod

            self.WeightsHidenp -= self.config.Learning_rate * (self.Update_Hiden_Batchp / float(self.data.SizeofTrain))
            self.Weights_Input -= self.config.Learning_rate * (self.Update_Input_Batch / float(self.data.SizeofTrain))
            self.Weights_Hiden -= self.config.Learning_rate * (self.Update_Hiden_Batch / float(self.data.SizeofTrain))
            self.Update_Hiden_Batchp = self.Update_Hiden_BatchpCopy.copy()
            self.Update_Input_Batch = np.zeros((self.data.X.shape[1], self.config.num_hidden_neurons_layer))
            self.Update_Hiden_Batch = np.zeros(self.config.num_hidden_neurons_layer)

            if (self.config.Learning_rate_schedule == ("True")):
                self.config.Learning_rate = self.config.Learning_rate * (1.0 / (1.0 + self.config.Decay * epoch))
        return

    def mini_batch_gradient_descent(self):
        self.batch = 0
        for epoch in range(self.config.num_epochs_training):
            for sample in range(self.data.SizeofTrain):
                for node in range(self.config.num_hidden_neurons_layer):
                    self.preActiv_Hp[0][node] = np.dot(self.data.Xtr[sample, :], self.Weights_Input[:, node])
                    self.postActiv_Hp[0][node] = self.activationfunction(self.preActiv_Hp[0][node], self.config.Type_activation_function)

                for layer in range(1, self.config.num_hidden_layers):
                    for node in range(self.config.num_hidden_neurons_layer):
                        self.preActiv_Hp[layer][node] = np.dot(self.postActiv_Hp[layer - 1], self.WeightsHidenp[layer - 1][:, node])
                        self.postActiv_Hp[layer][node] = self.activationfunction(self.preActiv_Hp[layer][node],self.config.Type_activation_function)

                preActivation_O = np.dot(self.postActiv_Hp[-1], self.Weights_Hiden)  # + Bias_Output)
                postActivation_O = self.activationfunction(preActivation_O, self.config.Type_activation_function)

                self.FE = self.lossfunction(postActivation_O, self.data.ytr[sample], self.config.Type_loss_function)

                for H_node in range(self.config.num_hidden_neurons_layer):
                    self.s_error = self.FE * self.activationfunctionp(preActivation_O, self.config.Type_activation_function)
                    self.gradient_HtoD = self.s_error * self.postActiv_Hp[-1][H_node]

                    # Before last hidden
                    if (self.config.num_hidden_layers > 1):
                        # Before last hidden
                        for I_node in range(self.config.num_hidden_neurons_layer):
                            input_value = self.postActiv_Hp[-2][I_node]
                            gradient_Itod = self.s_error * self.Weights_Hiden[H_node] * self.activationfunctionp(self.preActiv_Hp[-1][H_node], self.config.Type_activation_function) * input_value
                            self.s_errorp[-1][H_node] = self.activationfunctionp(self.preActiv_Hp[-1][H_node], self.config.Type_activation_function) * input_value
                            self.Update_Hiden_Batchp[-1][I_node, H_node] += gradient_Itod
                    else:
                        for I_node in range(self.data.X.shape[1]):
                            input_value = self.data.Xtr[sample, I_node]
                            self.gradient_Itod = self.s_error * self.Weights_Hiden[H_node] * self.activationfunctionp(self.preActiv_Hp[0][H_node],self.config.Type_activation_function) * input_value
                            self.Update_Input_Batch[I_node, H_node] += self.gradient_Itod

                    self.Update_Hiden_Batch[H_node] += self.gradient_HtoD

                if (self.config.num_hidden_layers > 1):
                    # Middle hiddens
                    for HidenLayer in range(self.config.num_hidden_layers - 3, -1, -1):
                        for H_node in range(self.config.num_hidden_neurons_layer):
                            for I_node in range(self.config.num_hidden_neurons_layer):
                                input_value = self.postActiv_Hp[HidenLayer][I_node]
                                self.s_errorp[HidenLayer][H_node] = self.activationfunctionp(self.preActiv_Hp[HidenLayer + 1][H_node],self.config.Type_activation_function) * input_value
                                self.gradient_Itod = np.dot(self.s_errorp[HidenLayer + 1],self.WeightsHidenp[HidenLayer + 1][H_node]) * self.s_errorp[HidenLayer][H_node]
                                self.Update_Hiden_Batchp[HidenLayer][I_node, H_node] += gradient_Itod

                        # First hidden
                    for H_node in range(self.config.num_hidden_neurons_layer):
                        for I_node in range(self.data.X.shape[1]):
                            input_value = self.data.Xtr[sample, I_node]
                            self.gradient_Itod = np.dot(self.s_errorp[0], self.WeightsHidenp[0][H_node]) * input_value
                            self.Update_Input_Batch[I_node, H_node] += gradient_Itod

                self.batch += 1
                if (self.batch >= self.config.Batch_size):
                    self.Weights_Input -= self.config.Learning_rate * (self.Update_Input_Batch / float(self.config.Batch_size))
                    self.Weights_Hiden -= self.config.Learning_rate * (self.Update_Hiden_Batch / float(self.config.Batch_size))
                    self.WeightsHidenp -= self.config.Learning_rate * (self.Update_Hiden_Batchp / float(self.config.Batch_size))
                    self.Update_Input_Batch = np.zeros((self.data.X.shape[1], self.config.num_hidden_neurons_layer))
                    self.Update_Hiden_Batch = np.zeros(self.config.num_hidden_neurons_layer)
                    self.Update_Hiden_Batchp = self.Update_Hiden_BatchpCopy.copy()
                    self.batch = 0
            if (self.config.Learning_rate_schedule == ("True")):
                self.config.Learning_rate = self.config.Learning_rate * (1.0 / (1.0 + self.config.Decay * epoch))
        return

    def verify(self):
        # Function to verify the neural network
        for sample in range(self.data.X.shape[0]-self.data.SizeofTrain):
            for node in range(self.config.num_hidden_neurons_layer):
                self.preActiv_Hp[0][node] = np.dot(self.data.Xt[sample, :], self.Weights_Input[:, node])
                self.postActiv_Hp[0][node] = self.activationfunction(self.preActiv_Hp[0][node], self.config.Type_activation_function)

            for layer in range(1, self.config.num_hidden_layers):
                for node in range(self.config.num_hidden_neurons_layer):
                    self.preActiv_Hp[layer][node] = np.dot(self.postActiv_Hp[layer - 1], self.WeightsHidenp[layer - 1][:, node])
                    self.postActiv_Hp[layer][node] = self.activationfunction(self.preActiv_Hp[layer][node], self.config.Type_activation_function)

            preActivation_O = np.dot(self.postActiv_Hp[-1], self.Weights_Hiden)
            postActivation_O = self.activationfunction(preActivation_O, self.config.Type_activation_function)

            self.FE2[sample] = postActivation_O

    def reset_array(self):
        # Function to reset arrays
        self.preActiv_Hp = np.empty(self.config.num_hidden_layers)
        self.preActiv_Hp = np.array(self.preActiv_Hp, dtype=object)
        self.postActiv_Hp = np.empty(self.config.num_hidden_layers)
        self.postActiv_Hp = np.array(self.postActiv_Hp, dtype=object)
        self.s_errorp = np.empty(self.config.num_hidden_layers)
        self.s_errorp = np.array(self.s_errorp, dtype=object)
        self.FE2 = np.empty(self.data.Xt.shape[0])
        self.FE2 = np.array(self.FE2)

        for i in range(self.config.num_hidden_layers):
            self.preActiv_Hp[i] = np.zeros(self.config.num_hidden_neurons_layer)
            self.postActiv_Hp[i] = np.zeros(self.config.num_hidden_neurons_layer)
            self.s_errorp[i] = np.zeros(self.config.num_hidden_neurons_layer)

    def lossfunction(self, y, r, o):
        if(o == 1):
            loss = y - r #Classic error
        if(o == 2):
            loss = (y - r)**2 #Squared Error
        if(o == 3):
            loss = ((y - r)**2)/2.0 #Variation of Squarred Error
        if(o == 4):
            loss = abs(y - r) #Absolute error
        return(loss)

    def activationfunction(self, Z, x):
        if(x == 1):
            act = self.sigmoid(Z)
        if(x == 2):
            act = math.tanh(Z)
        if(x == 3):
            act = self.ReLU(Z)
        return act

    def activationfunctionp(self, Z, x):
        if(x == 1):
            actp = self.sigmoidp(Z)
        if(x == 2):
            actp = self.hyperbolictangentp(Z)
        if(x == 3):
            actp = self.ReLUp(Z)
        return actp

    #ReLU
    def ReLU(self, Z):
        act = max(0, Z)
        return act

    def ReLUp(self, Z):
        actp = 0
        if(Z > 0):
            actp = 1
        return actp

    #Sigmoid function
    def sigmoid(self, Z):
        sig = (1.0/(1.0 + np.exp(-Z)))
        return sig
    
    def sigmoidp(self, Z):
        sigp = self.sigmoid(Z)*(1.0-self.sigmoid(Z))
        return sigp

    #Hyperbolic tangent
    def hyperbolictangentp(self, Z):
        htp = 1.0 - math.tanh(Z)**2.0
        return htp

    def TraceConfusionMatrix(self):
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for l1, l2 in zip(self.YActual, self.YPredicted):
            if (l1 == 1 and l2 == 1):
                TP = TP + 1
            elif (l1 == 0 and l2 == 0):
                TN = TN + 1
            elif (l1 == 1 and l2 == 0):
                FN = FN + 1
            elif (l1 == 0 and l2 == 1):
                FP = FP + 1

        print("Confusion Matrix: ")

        print("TP = ", TP)
        print("TN = ", TN)
        print("FP = ", FP)
        print("FN = ", FN)

        # Precision = TruePositives / (TruePositives + FalsePositives)
        # Recall = TruePositives / (TruePositives + FalseNegatives)
        if(TP + FP == 0):
            self.P = "cant be determined as all cases have been predicted to be negative" 
        else:
            self.P = TP / (TP + FP)
        if(TP + FN == 0):
            self.R = "no positive cases in the input data no conclusion can be made"
        else:
            self.R = TP / (TP + FN)

        print("Precision= ", self.P)
        print("Recall= ", self.R)

    def benchmark(self, index, data):
        if(index == 1):
            self.TraceGraph1(data)
        elif(index == 2):
            self.TraceGraph2(data)
        elif(index == 3):
            self.TraceGraph3(data)
        elif(index == 4):
            self.TraceGraph4(data)
        elif(index == 5):
            self.TraceGraph5(data)

    def TraceGraph1(self, data):
        #Test the different type of gradient descent
        num_iterations = 3  # Number of gradient descent types to test
        AccL = []
        TrainingTimeL = []
        Annotation = ["Stochastic gradient", "Batch gradient", "Mini batch gradient"]

        for i in tqdm(range(num_iterations), desc="Running Different Gradient Descent"):
            self.config.Type_gradient_descent = i + 1
            self.test(data)
            result = pd.DataFrame({'Result': self.FE2})
            TrainingTime = self.TrainingTime
            objective = pd.DataFrame({'Objective': data.yt})
            f = pd.DataFrame(np.around(result, decimals=6)).join(objective)
            f['pred'] = f['Result'].apply(lambda x: 0 if x < 0.5 else 1)
            Acc = f.loc[f['pred'] == f['Objective']].shape[0] / f.shape[0] * 100

            AccL.append(Acc)
            TrainingTimeL.append(TrainingTime)

        plt.title("Comparison of accuracy and training time \n for different types of gradient descent")
        plt.plot(AccL, TrainingTimeL, 'bP')
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AccL)):
            plt.annotate(Annotation[i], xy=(AccL[i], TrainingTimeL[i]))
        plt.show()

    def TraceGraph2(self, data):
        #Test different learning rate
        objective = pd.DataFrame({'Objective': data.yt})

        FELVL = []
        TrainingTimeLVL = []
        resultLVL = []
        fLVL = []
        AccLVL = []
        AnnotationLVL = []

        num_iterations = 10

        for i in tqdm(range(1, num_iterations + 1), desc="Testing Learning Rates"):
            self.config.Learning_rate = i / 10
            AnnotationLVL.append("Learning rate: " + str(self.config.Learning_rate))
            self.test(data)
            FELVL.append(self.FE2)
            TrainingTimeLVL.append(self.TrainingTime)
            resultLVL.append(pd.DataFrame({'Result' + str(i) + '': FELVL[i - 1]}))
            fLVL.append(pd.DataFrame(np.around(resultLVL[i - 1], decimals=6)).join(objective))
            fLVL[i - 1]['pred'] = fLVL[i - 1]['Result' + str(i)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVL.append(fLVL[i - 1].loc[fLVL[i - 1]['pred'] == fLVL[i - 1]['Objective']].shape[0] / fLVL[i - 1].shape[0] * 100)

        plt.title("Comparison of accuracy and training time for different learning rates")
        plt.plot(AccLVL, TrainingTimeLVL, 'bP')
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AccLVL)):
            plt.annotate(AnnotationLVL[i], xy=(AccLVL[i], TrainingTimeLVL[i]))
        plt.show()

    def TraceGraph3(self, data):
        #Test different number of epoch for learning rate and learning rate with decay
        objective = pd.DataFrame({'Objective': data.yt})

        FELVE = []
        TrainingTimeLVE = []
        resultLVE = []
        fLVE = []
        AccLVE = []
        AnnotationLVE = []

        FELVED = []
        TrainingTimeLVED = []
        resultLVED = []
        fLVED = []
        AccLVED = []
        AnnotationLVED = []

        FELVED2 = []
        TrainingTimeLVED2 = []
        resultLVED2 = []
        fLVED2 = []
        AccLVED2 = []
        AnnotationLVED2 = []

        self.config.Learning_rate = 0.3
        self.config.Learning_rate_schedule = False
        num_epochs_range = range(5, 50, 5)  # Adjust the range as needed

        for num_epochs in tqdm(num_epochs_range, desc="Testing Different number of epoch for Learning Rate = 0.3 without Decay"):
            self.config.num_epochs_training = num_epochs
            AnnotationLVE.append(str(num_epochs))
            self.test(data)
            FELVE.append(self.FE2)
            TrainingTimeLVE.append(self.TrainingTime)
            resultLVE.append(pd.DataFrame({'Result' + str(num_epochs) + '': FELVE[-1]}))
            fLVE.append(pd.DataFrame(np.around(resultLVE[-1], decimals=6)).join(objective))
            fLVE[-1]['pred'] = fLVE[-1]['Result' + str(num_epochs)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVE.append(fLVE[-1].loc[fLVE[-1]['pred'] == fLVE[-1]['Objective']].shape[0] / fLVE[-1].shape[0] * 100)

        self.config.Learning_rate_schedule = True
        self.config.Decay = 0.1

        for num_epochs in tqdm(num_epochs_range, desc="Testing Different number of epoch for Learning Rate = 0.3 with Decay = 0.1"):
            self.config.num_epochs_training = num_epochs
            AnnotationLVED.append(str(num_epochs))
            self.test(data)
            FELVED.append(self.FE2)
            TrainingTimeLVED.append(self.TrainingTime)
            resultLVED.append(pd.DataFrame({'Result' + str(num_epochs) + '': FELVED[-1]}))
            fLVED.append(pd.DataFrame(np.around(resultLVED[-1], decimals=6)).join(objective))
            fLVED[-1]['pred'] = fLVED[-1]['Result' + str(num_epochs)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVED.append(fLVED[-1].loc[fLVED[-1]['pred'] == fLVED[-1]['Objective']].shape[0] / fLVED[-1].shape[0] * 100)

        self.config.Decay = 0.4

        for num_epochs in tqdm(num_epochs_range, desc="Testing Different number of epoch for Learning Rate = 0.3 with Decay = 0.4"):
            self.config.num_epochs_training = num_epochs
            AnnotationLVED2.append(str(num_epochs))
            self.test(data)
            FELVED2.append(self.FE2)
            TrainingTimeLVED2.append(self.TrainingTime)
            resultLVED2.append(pd.DataFrame({'Result' + str(num_epochs) + '': FELVED2[-1]}))
            fLVED2.append(pd.DataFrame(np.around(resultLVED2[-1], decimals=6)).join(objective))
            fLVED2[-1]['pred'] = fLVED2[-1]['Result' + str(num_epochs)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVED2.append(fLVED2[-1].loc[fLVED2[-1]['pred'] == fLVED2[-1]['Objective']].shape[0] / fLVED2[-1].shape[0] * 100)

        plt.title("Comparison of accuracy and training time for different \n number of epochs for training between an MLP with a fixed learning rate \n and an MLP with a decaying learning rate")
        plt.plot(AccLVE, TrainingTimeLVE)
        plt.plot(AccLVED, TrainingTimeLVED)
        plt.plot(AccLVED2, TrainingTimeLVED2)
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AccLVE)):
            plt.annotate(AnnotationLVE[i], xy=(AccLVE[i], TrainingTimeLVE[i]))
            plt.annotate(AnnotationLVED[i], xy=(AccLVED[i], TrainingTimeLVED[i]))
            plt.annotate(AnnotationLVED2[i], xy=(AccLVED2[i], TrainingTimeLVED2[i]))
        plt.legend(["No decay", "Decay = 0.1", "Decay = 0.4"], loc="lower right")
        plt.show()
        self.config.Learning_rate_schedule = False

  
    def TraceGraph4(self, data):
        # Test the different activation function
        objective = pd.DataFrame({'Objective': data.yt})
        self.config.Learning_rate_schedule = False
        self.config.Learning_rate = 0.3

        activation_functions = [1, 2, 3]

        plt.figure(figsize=(10, 6))

        for activation_function in activation_functions:
            if activation_function == 1:
                activation_function_name = "Sigmoid"
            elif activation_function == 2:
                activation_function_name = "Hyperbolic tangent"
            elif activation_function == 3:
                activation_function_name = "ReLU"

            FELVA = []
            TrainingTimeLVA = []
            resultLVA = []
            fLVA = []
            AccLVA = []
            AnnotationLVA = []

            for i in tqdm(range(1, 15), desc=f"Testing {activation_function_name} Activation Function"):
                self.config.Type_activation_function = activation_function
                self.config.num_epochs_training = i * 3
                AnnotationLVA.append(f"{activation_function_name}, Epochs = {self.config.num_epochs_training}")
                self.test(data)
                FELVA.append(self.FE2)
                TrainingTimeLVA.append(self.TrainingTime)
                resultLVA.append(pd.DataFrame({'Result' + str(i) + '': FELVA[-1]}))
                fLVA.append(pd.DataFrame(np.around(resultLVA[-1], decimals=6)).join(objective))
                fLVA[-1]['pred'] = fLVA[-1]['Result' + str(i)].apply(lambda x: 0 if x < 0.5 else 1)
                AccLVA.append(fLVA[-1].loc[fLVA[-1]['pred'] == fLVA[-1]['Objective']].shape[0] / fLVA[-1].shape[0] * 100)

            plt.plot(AccLVA, TrainingTimeLVA, label=activation_function_name)

        plt.title("Comparison of accuracy and training time for different activation functions \n with the number of epochs for training varying")
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AnnotationLVA)):
                plt.annotate(AnnotationLVA[i], xy=(AccLVA[i], TrainingTimeLVA[i]))
        plt.legend(["Sigmoid", "Hyperbolic tangent", "ReLU"], loc="upper left")
        plt.show()
        self.config.Learning_rate_schedule = False


    def TraceGraph5(self, data):
        #Evolution with 1 and 2 hiden layer of the accuracy and training time
        self.config.Type_activation_function = 1
        objective = pd.DataFrame({'Objective': data.yt})

        FELVN1= []
        TrainingTimeLVN1 = []
        resultLVN1 = []
        fLVN1 = []
        AccLVN1 = []
        AnnotationLVN1 = []

        FELVN2= []
        TrainingTimeLVN2 = []
        resultLVN2 = []
        fLVN2 = []
        AccLVN2 = []
        AnnotationLVN2 = []

        plt.figure(figsize=(10, 6))

        self.config.num_hidden_layers = 2
        for i in tqdm(range(3, 10), desc="Testing 2 Hidden Layers"):
            self.config.num_hidden_neurons_layer = i * 2
            AnnotationLVN2.append("Neurons = " + str(self.config.num_hidden_neurons_layer))
            self.test(data)
            FELVN2.append(self.FE2)
            TrainingTimeLVN2.append(self.TrainingTime)
            resultLVN2.append(pd.DataFrame({'Result' + str(i - 2) + '': FELVN2[-1]}))
            fLVN2.append(pd.DataFrame(np.around(resultLVN2[-1], decimals=6)).join(objective))
            fLVN2[-1]['pred'] = fLVN2[-1]['Result' + str(i - 2)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVN2.append(fLVN2[-1].loc[fLVN2[-1]['pred'] == fLVN2[-1]['Objective']].shape[0] / fLVN2[-1].shape[0] * 100)

        self.config.num_hidden_layers = 1
        for i in tqdm(range(3, 10), desc="Testing 1 Hidden Layer"):
            self.config.num_hidden_neurons_layer = i * 2
            AnnotationLVN1.append("Neurons = " + str(self.config.num_hidden_neurons_layer))
            self.test(data)
            FELVN1.append(self.FE2)
            TrainingTimeLVN1.append(self.TrainingTime)
            resultLVN1.append(pd.DataFrame({'Result' + str(i - 2) + '': FELVN1[-1]}))
            fLVN1.append(pd.DataFrame(np.around(resultLVN1[-1], decimals=6)).join(objective))
            fLVN1[-1]['pred'] = fLVN1[-1]['Result' + str(i - 2)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVN1.append(fLVN1[-1].loc[fLVN1[-1]['pred'] == fLVN1[-1]['Objective']].shape[0] / fLVN1[-1].shape[0] * 100)

        plt.title("Comparison of accuracy and training time for an MLP \n with 1 hidden layer and an MLP with 2 hidden layers, \n with variation of the number of neurons per layer")
        plt.plot(AccLVN1, TrainingTimeLVN1)
        plt.plot(AccLVN2, TrainingTimeLVN2)
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AccLVN1)):
            plt.annotate(AnnotationLVN1[i], xy=(AccLVN1[i], TrainingTimeLVN1[i]))
            plt.annotate(AnnotationLVN2[i], xy=(AccLVN2[i], TrainingTimeLVN2[i]))
        plt.legend(["1 hidden layer", "2 hidden layers"], loc="upper left")
        plt.show()


