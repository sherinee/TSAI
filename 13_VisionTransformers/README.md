
Cats vs dogs training using ViT.   
  
Set up as done in the following blog,  
  
https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/  
  
batch_size = 64  
epochs = 20  
lr = 3e-5  
gamma = 0.7  
  
Transformations used:  
Resize  
RandomResizedCrop   
RandomHorizontalFlip  

Training logs for first 3 epochs: 

100%
313/313 [02:26<00:00, 2.40it/s]
Epoch : 1 - loss : 0.6966 - acc: 0.5068 - val_loss : 0.6927 - val_acc: 0.5000

100%
313/313 [02:24<00:00, 2.51it/s]
Epoch : 2 - loss : 0.6919 - acc: 0.5182 - val_loss : 0.6867 - val_acc: 0.5520

100%
313/313 [02:23<00:00, 2.56it/s]
Epoch : 3 - loss : 0.6829 - acc: 0.5595 - val_loss : 0.6779 - val_acc: 0.5758


