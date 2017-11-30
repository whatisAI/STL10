STL10 is a 10 class image dataset. https://cs.stanford.edu/~acoates/stl10/ 
Each image is RGB, with 96x96 pixels.

The first attempt to train a CNN from scratch, attained a classification accuracy on the test data  of approximately 0.6 over 30 epochs. More hyperparameter tuning can still be done, though. 

The second attempt to classify this data, is via transfer learning. Using  the first n layers from  VGG16, a network that was trained on imagenet. By adding one additional fully connected layer, and training for these weights, the accuracy arrives at apprxomately 0.7 over 10 epochs. Also, more hyper parameter tuning can be done. 
