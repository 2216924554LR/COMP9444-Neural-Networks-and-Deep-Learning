# COMP9444-Neural-Networks-and-Deep-Learning

## data_processing:
    Firstly, we remove urls and images from raw data and restore words before tokenising.
## network:
    Our network has one input which is fed to two different subnetworks. 
    The output is a tuple includes rating score and category score.
    
    ###rating net
    We use two bidirectional lstm layers to figure out the contextual information. 
        The input size is embedding size, which is 50. And we set the hidden nodes number to 100 in the  first lstm layer.
        In the second one, the input size is double size of the output is the first one and we hidden nodes number to 50.
        For reducing the computation load and making the net more robust, we use a maxpooling layer to make the embedding dim more compact by discarding some small numbers.
        Given sentence lengthvaries between different batches, some padding work is neccessary before we feed them dense layer.
        We pad them to 50 in dim2.
        Then the tensor flows through two dense layers, the output is size 1, which is the rating score.
        The activation function is relu and sigmoid.
        
    ###category net
    The net structure is exactly the same thing before the final layer.
        Pytorch has the ability to deal with the unnormalized tensors in multilabel classification.
        So we don't use any activation function in the final dense layer.
    
    ###loss_funcation
    we use BCELoss() for rating and CrossEntropyLoss() to category. And add them as the final loss.
        Considering category classification has the lower accuracy, we set the weight of CrossEntropyLoss() slightly higher.
        
    ###optimiser
    we use Adam and set lr to 0.0005.
    
## Summary: We run the code on both GPU and CPU. It takes about 3 minutes on GPU.
    The accuracy of rating is around 0.94
    The accuracy of category is around 0.83
    The total accuracy is around 0.78
    The final weighted score is around 83
        
    We tried some different parameters, but the final scores do not change a lot.
    The reason of low accuracy in categorary may be caused by we can not sum the two losses simply. 
    The weighted parameters in two net will influence each other.
