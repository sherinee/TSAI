Submitted by: Assignment Group 18

Link to Colab notebook: https://drive.google.com/file/d/1rv6C5XYtzA5Ka09n4g42ZFRZIahC4n3j/view?usp=sharing

**Objective:**
We aim to build a network that would recognize handwritten digits when provided as an image input and know how to add any number between 0 to 9 to the number predicted from the MNIST image

**Model architecture:**
The network we decided to build is designed as follows:

- Two convolutions with pooling applied to feature map
- Dropout layer
- Six fully connected layers leading to the two outputs - recognized digit & addition result
- Design choices we made while building the network:

**The network would have 2 main components: **
- a. For digit recognition 
- b. For adding the digit to the random number
Once we have the output from component (a), we would concatenate the one-hot encoded random number to the output
The final output would contain: a. A vector of 10 number giving the probabilities of the image being 0 to 9 b. A vector of 19 numbers giving the probabilities of the addition being 0 to 18
The summary of the network looks like this:
Model Summary

**The Network / Model - Loss Function**
We went ahead with the Negative Log Likelihood loss as that is known to give good results for comparing outputs for classification problems like this one.

Yes, we set it up as a classification problem mainly because: a. Digit recognition is involved b. The summations are constrained within the range of 0 to 18, with the input combinations being finite

**The training**
Inspired by YOLO, we wanted to train the network for digit recognition for the first few epochs and then after it achieves certain level of accuracy, train it for the addition / summation objective

ACCURACY ACHIEVED - 97.75%
To our surprise, the network does not take more than a few epochs to achieve an ~ 98% accuracy

Here are the loss and accuracy charts from training the network:

Training Logs
Training Logs Screenshot

Training Loss By Epochs
Chart - Training Losses by Epochs

Digit Recognition Accuracy By Epochs
Chart - Digit Recognition Accuracy by Epochs

Addition Accuracy By Epochs
Chart - Addition Accuracy by Epochs

Training Loss By Batch Iterations
Chart - Training Losses by Epochs
