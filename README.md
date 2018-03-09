# STL10
STL10 is a 10 class image dataset. 
Download binary training and test files from https://cs.stanford.edu/~acoates/stl10/ Each image is RGB, with 96x96 pixels.

The training data set consists of 5000 images.


I tried two classification approaches with this dataset:   1) Train a CNN from scratch, and 2) Do transfer learning, by using the weights of VGG-16. The transfer learning provided a better result in this case. 
    
Considering we have 10 classes, we should at least have 0.1 accuracy.

The first attempt to train a CNN from scratch, attained a classification accuracy on the test data of approximately 0.6 over 30 epochs. More hyperparameter tuning can still be done, as we can see that there is some overfitting, because the training loss decreases while the test loss starts to increase. In addition, some work can still be done on the learning rate, as we can see the accuracy decreasing very slowly. All the details are in the notebook stl10_supervised_learning_vf2.ipynb

The second attempt to classify this data, is via transfer learning. Using the first n layers from VGG16, a network that was trained on imagenet. By adding one additional fully connected layer, and training for these weights, the accuracy arrives at apprxomately 0.7 over 10 epochs. Also, more hyper parameter tuning can be done. All the details are in the notebook stl10_transfer_learning_vf.ipynb.

Finally, following the ideas in https://cs.stanford.edu/~acoates/stl10/, as a next step it would be worth-while to use the 100,000 unlabeled images via what is known as semi supervised learning, assuming that they come from a similar distribution as the images in the training set. A possible way to do this would be to start by training an autoencoder , like what is nicely explained in this Keras tutorial https://blog.keras.io/building-autoencoders-in-keras.html. The idea would be to train an autoencoder that would be a low-level or sparse feature representation of the unlabled images. Then, pass all the training images through the auto encoder before training the CNN. This may, or may not, improve either the accuracy or the decrease the training time.
