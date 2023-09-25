class Config:
    def __init__(self):
        #Parameter
        self.num_hidden_layers = -1 #Doesn't wortk very well for more than 2 hidden layer
        self.num_hidden_neurons_layer = -1
        self.Type_activation_function = -1
        self.Type_loss_function = -1
        self.num_epochs_training = -1
        self.Learning_rate = -1
        self.Learning_rate_schedule = False
        self.Decay = -1
        self.Type_gradient_descent = -1
        self.Batch_size = -1

    def ask_parameters(self):
        # Function to get user input for configuration parameters
        while(self.num_hidden_layers < 0):
            try:
                self.num_hidden_layers = int(input("How many hidden layers do you want to work with (more than 2 hiden layers makes the program unstable):"))
            except ValueError:
                print("Input a valid choice please")
        while(self.num_hidden_neurons_layer < 0):
            try:
                self.num_hidden_neurons_layer = int(input("How many neurons do you want per hidden layer ?:"))
            except ValueError:
                print("Input a valid choice please")
        while(self.Type_activation_function<1 or self.Type_activation_function>3):
            try:
                self.Type_activation_function = int(input("Chose you activation function type (1: Sigmoid 2: Hyperbolic tangent 3: ReLU):"))
            except ValueError:
                print("Input a valid choice please")
        while (self.num_epochs_training < 0):
            try:
                self.num_epochs_training = int(input("Chose your number of epochs for the training:"))
            except ValueError:
                print("Input a valid choice please")
        while(self.Learning_rate < 0):
            try:
                self.Learning_rate = float(input("Chose your learning rate:"))
            except ValueError:
                print("Input a valid choice please")
        while(True):
            try:
                self.Learning_rate_schedule = bool(input("Do you want to activate a learning rate schedule (0: No 1: Yes):"))
                break
            except ValueError:
                print("Input a valid choice please")
        if(self.Learning_rate_schedule == ("True")):
            while(self.Decay < 0):
                try:
                    self.Decay = float(input("Chose your decay constant:"))
                except ValueError:
                    print("Input a valid choice please")
        while(self.Type_loss_function < 1 or self.Type_loss_function > 4):
            try:
                self.Type_loss_function = int(input("Chose your loss function (1: Normal 2: Squared error 3: Squarred error/2 4: Absolute error):"))
                break
            except ValueError:
                print("Input a valid choice please")
        while(self.Type_gradient_descent<1 or self.Type_gradient_descent>3):
            try:
                self.Type_gradient_descent = int(input("Chose your type of gradient descent (1: Stochastic 2: Batch 3: MiniBatch):"))
                break
            except ValueError:
                print("Input a valid choice please")
        if(self.Type_gradient_descent == 3):
            while (self.Batch_size < 0 ):
                try:
                    self.Batch_size = int(input("Chose the size of your mini batch:"))
                except ValueError:
                    print("Input a valid choice please")


