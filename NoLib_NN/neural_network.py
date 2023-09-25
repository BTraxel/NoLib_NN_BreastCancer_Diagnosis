import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.set_random_weights()
        self.reset_array()
        self.trained = False

    def set_random_weights(self):
        # Function to initialize neural network weights
        self.Weights_Input = np.random.uniform(-1, 1, (self.data.X.shape[1], self.config.num_hidden_neurons_layer))* np.sqrt(2 / self.data.X.shape[1])
        self.Update_Input_Batch = np.zeros((self.data.X.shape[1], self.config.num_hidden_neurons_layer))
        self.Weights_Hiden = np.random.uniform(-1, 1, self.config.num_hidden_neurons_layer)* np.sqrt(2 / self.config.num_hidden_neurons_layer)
        self.Update_Hiden_Batch = np.zeros(self.config.num_hidden_neurons_layer)
        self.WeightsHidenp = np.empty(self.config.num_hidden_layers - 1)
        self.WeightsHidenp = np.array(self.WeightsHidenp, dtype=object)
        self.Update_Hiden_Batchp = np.empty(self.config.num_hidden_layers - 1)
        self.Update_Hiden_Batchp = np.array(self.Update_Hiden_Batchp, dtype=object)

        for i in range(self.config.num_hidden_layers-1):
            self.WeightsHidenp[i] = np.random.uniform(-1, 1, (self.config.num_hidden_neurons_layer, self.config.num_hidden_neurons_layer))* np.sqrt(2 / self.config.num_hidden_neurons_layer)
            self.Update_Hiden_Batchp[i] = np.zeros((self.config.num_hidden_neurons_layer, self.config.num_hidden_neurons_layer))

        self.Update_Hiden_BatchpCopy = self.Update_Hiden_Batchp.copy()
        return

    def train(self):
        # Function to train the neural network
        if (self.config.Type_gradient_descent == 1):  # Stochastic
            self.stochastic_gradient_descent()
        
        elif (self.config.Type_gradient_descent == 2):  # Batch
            self.batch_gradient_descent()
        
        elif (self.config.Type_gradient_descent == 3):  # Mini-batch
            self.mini_batch_gradient_descent()
        self.trained = True
        return

    
    def test(self):
        if(self.trained):
            self.reset()
            self.trained = False
        self.start_time = time.time()
        self.train()
        self.TrainingTime = time.time() - self.start_time
        self.verify()
        return

    def simpleBenchmark(self):
        self.test()
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
        return

    def reset(self):
        self.set_random_weights
        self.reset_array()
        return
    
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
        return

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

    def benchmark(self):
        self.TraceGraph1()
        self.TraceGraph2()
        self.TraceGraph3()
        self.TraceGraph4()
        self.TraceGraph5()

    def TraceGraph1(self):

        self.config.Type_gradient_descent = 1
        self.test()
        result1 = pd.DataFrame({'Result1': self.FE2})
        TrainingTime1 = self.TrainingTime

        self.config.Type_gradient_descent = 2
        self.test()
        result2 = pd.DataFrame({'Result2': self.FE2})
        TrainingTime2 = self.TrainingTime

        self.config.Type_gradient_descent = 3
        self.test()
        result3 = pd.DataFrame({'Result3': self.FE2})
        TrainingTime3 = self.TrainingTime

        objective = pd.DataFrame({'Objective': self.data.yt})
        f1 = pd.DataFrame(np.around(result1, decimals=6)).join(objective)
        f2 = pd.DataFrame(np.around(result2, decimals=6)).join(objective)
        f3 = pd.DataFrame(np.around(result3, decimals=6)).join(objective)
        f1['pred'] = f1['Result1'].apply(lambda x: 0 if x < 0.5 else 1)
        f2['pred'] = f2['Result2'].apply(lambda x: 0 if x < 0.5 else 1)
        f3['pred'] = f3['Result3'].apply(lambda x: 0 if x < 0.5 else 1)
        Acc1 = f1.loc[f1['pred'] == f1['Objective']].shape[0] / f1.shape[0] * 100
        Acc2 = f2.loc[f2['pred'] == f2['Objective']].shape[0] / f2.shape[0] * 100
        Acc3 = f3.loc[f3['pred'] == f3['Objective']].shape[0] / f3.shape[0] * 100

        AccL = [Acc1,Acc2,Acc3]
        TrainingTimeL = [TrainingTime1,TrainingTime2,TrainingTime3]
        Annotation = ["Stochastic gradient","Batch gradient","Mini batch gradient"]

        plt.title("Comparison of accuracy and training time \n for different type of gradient descent")
        plt.plot(AccL,TrainingTimeL,'bP')
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AccL)):
            plt.annotate(Annotation[i], xy=(AccL[i], TrainingTimeL[i]))
        plt.show()

    def TraceGraph2(self):
        objective = pd.DataFrame({'Objective': self.data.yt})

        FELVL = []
        TrainingTimeLVL = []
        resultLVL = []
        fLVL = []
        AccLVL = []
        AnnotationLVL = []
        for i in range(1, 11):
            self.config.Learning_rate = i/10
            AnnotationLVL.append("Learning rate: "+str(self.config.Learning_rate))
            self.test()
            FELVL.append(self.FE2)
            TrainingTimeLVL.append(self.TrainingTime)
            resultLVL.append(pd.DataFrame({'Result'+str(i)+'': FELVL[i-1]}))
            fLVL.append(pd.DataFrame(np.around(resultLVL[i-1], decimals=6)).join(objective))
            fLVL[i-1]['pred'] = fLVL[i-1]['Result'+str(i)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVL.append(fLVL[i-1].loc[fLVL[i-1]['pred'] == fLVL[i-1]['Objective']].shape[0] / fLVL[i-1].shape[0] * 100)

        plt.title("Comparison of accuracy and training time for different learning rates")
        plt.plot(AccLVL, TrainingTimeLVL,'bP')
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AccLVL)):
            plt.annotate(AnnotationLVL[i], xy=(AccLVL[i], TrainingTimeLVL[i]))
        plt.show()

    def TraceGraph3(self):
        #Test different number of epoch for learning rate and learning rate with decay
        objective = pd.DataFrame({'Objective': self.data.yt})

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
        for i in range(1, 10):
            self.config.num_epochs_training = i * 5
            AnnotationLVE.append(str(self.config.num_epochs_training))
            self.test()
            FELVE.append(self.FE2)
            TrainingTimeLVE.append(self.TrainingTime)
            resultLVE.append(pd.DataFrame({'Result'+str(i)+'': FELVE[i-1]}))
            fLVE.append(pd.DataFrame(np.around(resultLVE[i-1], decimals=6)).join(objective))
            fLVE[i-1]['pred'] = fLVE[i-1]['Result'+str(i)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVE.append(fLVE[i-1].loc[fLVE[i-1]['pred'] == fLVE[i-1]['Objective']].shape[0] / fLVE[i-1].shape[0] * 100)

        self.config.Learning_rate_schedule = True
        self.config.Decay = 0.1
        for i in range(1, 10):
            self.config.num_epochs_training = i * 5
            AnnotationLVED.append(str(self.config.num_epochs_training))
            self.test()
            FELVED.append(self.FE2)
            TrainingTimeLVED.append(self.TrainingTime)
            resultLVED.append(pd.DataFrame({'Result' + str(i) + '': FELVED[i - 1]}))
            fLVED.append(pd.DataFrame(np.around(resultLVED[i - 1], decimals=6)).join(objective))
            fLVED[i - 1]['pred'] = fLVED[i - 1]['Result' + str(i)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVED.append(fLVED[i - 1].loc[fLVED[i - 1]['pred'] == fLVED[i - 1]['Objective']].shape[0] / fLVED[i - 1].shape[0] * 100)

        self.config.Decay = 0.4
        for i in range(1, 10):
            self.config.num_epochs_training = i * 5
            AnnotationLVED2.append(str(self.config.num_epochs_training))
            self.test()
            FELVED2.append(self.FE2)
            TrainingTimeLVED2.append(self.TrainingTime)
            resultLVED2.append(pd.DataFrame({'Result' + str(i) + '': FELVED2[i - 1]}))
            fLVED2.append(pd.DataFrame(np.around(resultLVED2[i - 1], decimals=6)).join(objective))
            fLVED2[i - 1]['pred'] = fLVED2[i - 1]['Result' + str(i)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVED2.append(fLVED2[i - 1].loc[fLVED2[i - 1]['pred'] == fLVED2[i - 1]['Objective']].shape[0] / fLVED2[i - 1].shape[0] * 100)


        plt.title("Comparison of accuracy and training time for different \n number of epoch for training between an MLP with a fixed learning rate \n and an MLP with a decaying learning rate")
        plt.plot(AccLVE, TrainingTimeLVE)
        plt.plot(AccLVED, TrainingTimeLVED)
        plt.plot(AccLVED2, TrainingTimeLVED2)
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AccLVE)):
            #plt.annotate(AnnotationLVE[i], xy=(AccLVE[i], TrainingTimeLVE[i]))
            plt.annotate(AnnotationLVED[i], xy=(AccLVED[i], TrainingTimeLVED[i]))
        plt.legend(["No decay", "Decay = 0.1", "Decay = 0.4"], loc="lower right")
        plt.show()
        self.config.Learning_rate_schedule = False

    def TraceGraph4(self):
        # Test the different activation function = 1
        #Test different number of epoch for learning rate and learning rate with decay
        objective = pd.DataFrame({'Objective': self.data.yt})
        self.config.Learning_rate_schedule = False
        self.config.Learning_rate = 0.3
        FELVA1 = []
        TrainingTimeLVA1 = []
        resultLVA1 = []
        fLVA1 = []
        AccLVA1 = []
        AnnotationLVA1 = []

        FELVA2 = []
        TrainingTimeLVA2 = []
        resultLVA2 = []
        fLVA2 = []
        AccLVA2 = []
        AnnotationLVA2 = []

        FELVA3 = []
        TrainingTimeLVA3 = []
        resultLVA3 = []
        fLVA3 = []
        AccLVA3 = []
        AnnotationLVA3 = []

        self.config.Type_activation_function = 1
        for i in range(1, 15):
            self.config.num_epochs_training = i * 3
            AnnotationLVA1.append(str(self.config.num_epochs_training))
            self.test()
            FELVA1.append(self.FE2)
            TrainingTimeLVA1.append(self.TrainingTime)
            resultLVA1.append(pd.DataFrame({'Result' + str(i) + '': FELVA1[i - 1]}))
            fLVA1.append(pd.DataFrame(np.around(resultLVA1[i - 1], decimals=6)).join(objective))
            fLVA1[i - 1]['pred'] = fLVA1[i - 1]['Result' + str(i)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVA1.append(fLVA1[i - 1].loc[fLVA1[i - 1]['pred'] == fLVA1[i - 1]['Objective']].shape[0] / fLVA1[i - 1].shape[0] * 100)

        self.config.Type_activation_function = 2
        for i in range(1, 15):
            self.config.num_epochs_training = i * 3
            AnnotationLVA2.append(str(self.config.num_epochs_training))
            FELVA2.append(self.FE2)
            TrainingTimeLVA2.append(self.TrainingTime)
            resultLVA2.append(pd.DataFrame({'Result' + str(i) + '': FELVA2[i - 1]}))
            fLVA2.append(pd.DataFrame(np.around(resultLVA2[i - 1], decimals=6)).join(objective))
            fLVA2[i - 1]['pred'] = fLVA2[i - 1]['Result' + str(i)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVA2.append(fLVA2[i - 1].loc[fLVA2[i - 1]['pred'] == fLVA2[i - 1]['Objective']].shape[0] / fLVA2[i - 1].shape[0] * 100)

        self.config.Type_activation_function = 3
        for i in range(1, 15):
            self.config.num_epochs_training = i * 3
            AnnotationLVA3.append("Ept = " + str(self.config.num_epochs_training))
            FELVA3.append(self.FE2)
            TrainingTimeLVA3.append(self.TrainingTime)
            resultLVA3.append(pd.DataFrame({'Result' + str(i) + '': FELVA3[i - 1]}))
            fLVA3.append(pd.DataFrame(np.around(resultLVA3[i - 1], decimals=6)).join(objective))
            fLVA3[i - 1]['pred'] = fLVA3[i - 1]['Result' + str(i)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVA3.append(fLVA3[i - 1].loc[fLVA3[i - 1]['pred'] == fLVA3[i - 1]['Objective']].shape[0] / fLVA3[i - 1].shape[0] * 100)

        plt.title("Comparison of accuracy and training time for different activation function \n with the number of epoch for training varying")
        plt.plot(AccLVA1, TrainingTimeLVA1)
        plt.plot(AccLVA2, TrainingTimeLVA2)
        plt.plot(AccLVA3, TrainingTimeLVA3)
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AccLVA1)):
            plt.annotate(AnnotationLVA1[i], xy=(AccLVA1[i], TrainingTimeLVA1[i]))
            plt.annotate(AnnotationLVA2[i], xy=(AccLVA2[i], TrainingTimeLVA2[i]))
            plt.annotate(AnnotationLVA3[i], xy=(AccLVA3[i], TrainingTimeLVA3[i]))
        plt.legend(["Sigmoid", "Hyperbolic tangent", "ReLU"], loc="upper left")
        plt.show()

    def TraceGraph5(self):
        #Evolution with 1 and 2 hiden layer of the accuracy and training time
        self.config.Type_activation_function = 1
        objective = pd.DataFrame({'Objective': self.data.yt})

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

        self.config.num_hidden_layers = 2
        for i in range(3, 10):
            self.config.num_hidden_neurons_layer = i * 2
            AnnotationLVN2.append("Neurons = " + str(self.config.num_hidden_neurons_layer))
            self.test()
            FELVN2.append(self.FE2)
            TrainingTimeLVN2.append(self.TrainingTime)
            resultLVN2.append(pd.DataFrame({'Result' + str(i - 2) + '': FELVN2[i - 3]}))
            fLVN2.append(pd.DataFrame(np.around(resultLVN2[i - 3], decimals=6)).join(objective))
            fLVN2[i - 3]['pred'] = fLVN2[i - 3]['Result' + str(i - 2)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVN2.append(fLVN2[i - 3].loc[fLVN2[i - 3]['pred'] == fLVN2[i - 3]['Objective']].shape[0] / fLVN2[i - 3].shape[0] * 100)

        self.config.num_hidden_layers = 1
        for i in range(3, 10):
            self.config.num_hidden_neurons_layer = i * 2
            AnnotationLVN1.append("Neurons = " + str(self.config.num_hidden_neurons_layer))
            self.test()
            FELVN1.append(self.FE2)
            TrainingTimeLVN1.append(self.TrainingTime)
            resultLVN1.append(pd.DataFrame({'Result' + str(i-2) + '': FELVN1[i - 3]}))
            fLVN1.append(pd.DataFrame(np.around(resultLVN1[i - 3], decimals=6)).join(objective))
            fLVN1[i - 3]['pred'] = fLVN1[i - 3]['Result' + str(i-2)].apply(lambda x: 0 if x < 0.5 else 1)
            AccLVN1.append(fLVN1[i - 3].loc[fLVN1[i - 3]['pred'] == fLVN1[i - 3]['Objective']].shape[0] / fLVN1[i - 3].shape[0] * 100)



        plt.title("Comparison of accuracy and training time for an MLP \n with 1 hiden layer and an MLP with 2 hiden layer, \n 1with variation of the number of neurons per layer")
        plt.plot(AccLVN1, TrainingTimeLVN1)
        plt.plot(AccLVN2, TrainingTimeLVN2)
        plt.ylabel("Training time")
        plt.xlabel("Accuracy")
        for i in range(len(AccLVN1)):
            plt.annotate(AnnotationLVN1[i], xy=(AccLVN1[i], TrainingTimeLVN1[i]))
            plt.annotate(AnnotationLVN2[i], xy=(AccLVN2[i], TrainingTimeLVN2[i]))
        plt.legend(["1 hidden layer", "2 hidden layer"], loc="upper left")
        plt.show()


