Goal is to create a classification model to predict digits with high accuracy and low number of parameters

Dataset used: MNIST (handwritten digit dataset)

Step 1: Split data into training, testing and validation folders 
Step 2: Develop the model with the required number of layers  

   ** **Layers used**  **

    - Convolution layers: For the model to learn edges, gradients, patterns, parts of objects and objects  
    - ReLu: The best activation function ever!   
    - MaxPool: To select important features from image  
    - Kernel used: 3 x 3 which is the standard kernel used extensively  
    - Batch normalization: to normalize within its batch  
    - DropOut layer: to force all the neurons to learn and avoid overfitting  
    - Loss function: Cross entropy loss  
    - Optimizer: Adam  
    - log_softmax: Preferred activation function for multi class classification  
    
Step 3: Repeat the process in different batches until the loss value and accuracy reaches a saturation point (In our case, I have stopped arbitrarily at 20)  
Step 4: Predict on test dataset
