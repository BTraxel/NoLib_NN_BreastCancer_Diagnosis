import numpy as np
import pandas as pd

class Data:
    def __init__(self):
        self.Splitpercent = 0

    def split_train_test(self):
        while(self.Splitpercent < 1 or self.Splitpercent > 100):
            try:
                self.Splitpercent = int(input("Give a percentage to divide dataset into Train and Test set:"))
            except ValueError:
                print("Input a valid choice please")

        self.SizeofTrain = int((self.Splitpercent/100)*self.X.shape[0])

        self.Xtrraw,self.Xtraw = np.array_split(self.X, [self.SizeofTrain])
        self.ytrraw,self.ytraw = np.array_split(self.y, [self.SizeofTrain])

        self.Xtr = np.asarray(self.Xtrraw)
        self.Xt = np.asarray(self.Xtraw)
        self.ytr = np.asarray(self.ytrraw)
        self.yt = np.asarray(self.ytraw)

    def get_data(self):
        # Function to load and preprocess data
        data = pd.read_csv("../Data.csv")
        print("Dataset size")
        print("Rows {} Columns {}".format(data.shape[0], data.shape[1]))
        print("Columns and data types")

        Size = int(data.shape[1]) -2
        print("There is ",Size," array of data that can be use to predict wheter the tumor is begnin or malignant")
        Used = -1
        while(Used < 1 or Used > Size):
            try:
                print("How many of them do you want to use (number between 1 and ",Size,"):")
                Used = int(input())
            except ValueError:
                print("Input a valid choice please")

        df = pd.DataFrame(data)
        # Convert Begnin Malignant feature from string to integer
        df['class'] = df.iloc[:, 1].apply(lambda x: 1.0 if x == "B" else 0.0)
        # Normalisation
        for i in df.columns:
            if ((i != "842302") & (i != "M")):
                df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())

        # features will be saved as X and our target will be saved as y
        SelectArray = []
        for u in range(0, Used):
            SelectArray.append(u+2)
        self.X = df.iloc[:, SelectArray].copy()
        self.y = df['class'].copy()
