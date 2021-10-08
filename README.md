# Domain-Guided-Dropout
- The model has been trained on MNIST dataset (only 15000 samples) and the performance is measure on all the 10000 test images with and without the presence of dropout
- The Model is randomly initialized and the paramters are used to predict s_i and m_i values for the training set
- The learning determinsitic m_i values for the last ff layer determine the dropout
- Test Performance :
  a. With Dropout : 96%
  b. Without Dropout : 92%
