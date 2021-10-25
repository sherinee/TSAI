**Filename: EVAI_S5_Code1.ipynb**  
Target:   
1) Set up a basic network   
  
Results:
1) Parameters: 8.6k  
2) Best Train Accuracy: 99.9  
3) Best Test Accuracy: 99.3    

Analysis: Performs well, need to increase accuracy further
  
    
**Filename: EVAI_S5_Code2.ipynb**  
Target:  
1) Improve accuracy further  
2) Added batch normalization
3) Removed the big kernels in the last layer -> replaced with max pooling layers 

Results:
1) Parameters: 12K
2) Best Train Accuracy:99.5  
3) Best Test Accuracy: 99.76  

Analysis: Accuracy target met, need to reduce number of parameters
  
    
    
**Filename: EVAI_S5_Code3.ipynb**    
Target:  
1) Reduce number of params  
2) Play around with number of neurons in each layer   

Results:
1) Parameters: 7.69K
2) Best Train Accuracy:99.21 (15th epoch)
3) Best Test Accuracy: 99.5 (15th epoch) 

Analysis: Accuracy and params target met, play around a bit more to stabilize the results and finalize  
  
    
    
**Filename: EVAI_S5_Code4.ipynb**  
Target:  
1) Stablize the accuracy  

Results:
1) Parameters: 8.6K
2) Best Train Accuracy:99.3 (14th epoch)
3) Best Test Accuracy: 99.64 (14th epoch) 
  
    
    
**Analysis: Model performs well. Test Accuracy remains greater than 99.4 in the last 5 epochs.**  

