#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

"""
data_processing:
    Firstly, we remove urls and images from raw data and restore words before tokenising.

network:
    Our network has one input which is fed to two different subnetworks. 
    The output is a tuple includes rating score and category score.
    
    rating net: We use two bidirectional lstm layers to figure out the contextual information. 
        The input size is embedding size, which is 50. And we set the hidden nodes number to 100 in the  first lstm layer.
        In the second one, the input size is double size of the output is the first one and we hidden nodes number to 50.
        For reducing the computation load and making the net more robust, we use a maxpooling layer to make the embedding dim more compact by discarding some small numbers.
        Given sentence lengthvaries between different batches, some padding work is neccessary before we feed them dense layer.
        We pad them to 50 in dim2.
        Then the tensor flows through two dense layers, the output is size 1, which is the rating score.
        The activation function is relu and sigmoid.
        
    category net: The net structure is exactly the same thing before the final layer.
        Pytorch has the ability to deal with the unnormalized tensors in multilabel classification.
        So we don't use any activation function in the final dense layer.
    
    loss_funcation: we use BCELoss() for rating and CrossEntropyLoss() to category. And add them as the final loss.
        Considering category classification has the lower accuracy, we set the weight of CrossEntropyLoss() slightly higher.
        
    optimiser: we use Adam and set lr to 0.0005.

    Summary: We run the code on both GPU and CPU. It takes about 3 minutes on GPU.
        The accuracy of rating is around 0.94
        The accuracy of category is around 0.83
        The total accuracy is around 0.78
        The final weighted score is around 83
        
        We tried some different parameters, but the final scores do not change a lot.
        The reason of low accuracy in categorary may be caused by we can not sum the two losses simply. 
        The weighted parameters in two net will influence each other.
        

"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as f
# import numpy as np
# import sklearn

import nltk
from nltk.corpus import stopwords
import re

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################
def apply_stopwords(text, sw):
    text_list = text.split()
    res = ''
    for word in text_list:
        if word.lower() not in sw:
            res = res + word + ' '
    return res.strip()

def clean_text(text):
    
    
    # Convert words to lower case and split them
    text = text.lower()
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", text)
    
    # Remove images
    text = re.sub(r"image:[a-zA-Z0-9]*\.jpg", " ", text)
    text = re.sub(r"image:[a-zA-Z0-9]*\.png", " ", text)
    text = re.sub(r"image:[a-zA-Z0-9]*\.gif", " ", text)
    text = re.sub(r"image:[a-zA-Z0-9]*\.bmp", " ", text)
        
    
    # Restore word
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    sw = stopwords.words('english')
    text = apply_stopwords(text, sw)
    
    # Return a list of words
    return text



def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    sample = clean_text(sample)
    processed = sample.split()

    return processed



def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    #sample = pad_sequences(sample, maxlen=50)

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch


stopWords = {}
wordVectors = GloVe(name='6B', dim=50)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    zero = torch.zeros_like(ratingOutput)
    one = torch.ones_like(ratingOutput)
    ratingOutput = torch.where(ratingOutput>=0.5, one, ratingOutput)
    ratingOutput = torch.where(ratingOutput<0.5, zero, ratingOutput)
    #print(ratingOutput)
    
    category_max = torch.max(categoryOutput, 1)[1]
    #print(categoryOutput)
    return ratingOutput, category_max

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        # input size (batch_size_32, sentence_length, embedding_size_50)
        self.lstm1 = tnn.LSTM(input_size=50,
                             hidden_size=100,
                             bidirectional=True,
                             batch_first=True,
                             dropout=0.1)

        self.lstm1_1 = tnn.LSTM(input_size=200,
                             hidden_size=50,
                             bidirectional=True,
                             batch_first=True,
                             dropout=0.1)
        
        self.lstm2 = tnn.LSTM(input_size=50,
                             hidden_size=100,
                             bidirectional=True,
                             batch_first=True,
                             dropout=0.1)
        
        self.lstm3 = tnn.LSTM(input_size=200,
                             hidden_size=50,
                             bidirectional=True,
                             batch_first=True,
                             dropout=0.1)
       
        self.dropout_layer_1 = tnn.Dropout(p=0.1)
        self.dropout_layer_2 = tnn.Dropout(p=0.1)
        
        self.linear_1 = tnn.Linear(800 ,512)
        self.linear_2 = tnn.Linear(512, 1)
        
        self.linear_3 = tnn.Linear(800, 512)
        self.linear_4 = tnn.Linear(512, 5)
        

    def forward(self, inputs, length):
        
        def pad(inputs, d):
            if inputs.size()[1] >= 40:
                inputs = inputs[:, :40, :]
            else:
                pad_tensor = torch.tensor([0]*32*(40 - inputs.size()[1])*d).to(device)
                pad_tensor = pad_tensor.reshape(32, -1, d)
                inputs = torch.cat((inputs, pad_tensor), 1)
            return inputs
        
        
        
        x = inputs
        
        # rating net
        x_r, _ = self.lstm1(x)
        x_r, (hidden, cell) = self.lstm1_1(x_r)
        x_r = f.max_pool1d(x_r, kernel_size=5)
        #print(x_r.shape)
        x_r = pad(x_r, 20)
        x_r = x_r.view(x_r.size()[0], 800)
        x_r = torch.relu(self.linear_1(x_r))
        x_r = self.dropout_layer_1(x_r)
        x_r = torch.sigmoid(self.linear_2(x_r))
        x_r = x_r.view(-1)
        
        # category net
        x_c, (hidden, cell) = self.lstm2(x)
        x_c, (hidden, cell) = self.lstm3(x_c)
        x_c = f.max_pool1d(x_c, kernel_size=5)
        x_c = pad(x_c, 20)
        x_c = x_c.view(x_c.size()[0], 800)
        x_c = torch.relu(self.linear_3(x_c))
        x_c = self.dropout_layer_2(x_c)
        x_c = self.linear_4(x_c)
        
        return x_r, x_c
    

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.BlossFunc = tnn.BCELoss().to(device)
        self.ClossFunc = tnn.CrossEntropyLoss().to(device)
        
    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        rate_loss = self.BlossFunc(ratingOutput, ratingTarget.float())
        cate_loss = self.ClossFunc(categoryOutput, categoryTarget)
        
        return 0.4*rate_loss + 0.6*cate_loss
    


################################################################################
################## The following determines training options ###################
################################################################################
nltk.download('stopwords')
trainValSplit = 0.8
batchSize = 32
epochs = 7 
net = network()
lossFunc = loss()
optimiser = toptim.Adam(net.parameters(), lr=0.0005)

