import numpy as np
from numpy.lib.function_base import gradient
import tensorflow as tf
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CNN_Block(keras.layers.Layer):
    def __init__(self, cnn_layers, filters, kernel ,**kwargs):
        super().__init__(**kwargs)
        self.cnns = [keras.layers.Conv2D(filters,kernel,activation="relu", padding = "same", kernel_initializer="he_normal") for i in range(cnn_layers)]
        self.max_pool = keras.layers.MaxPool2D(pool_size=2, padding = "same")

    def call(self, inputs):
        z = inputs
        # print(z.shape)
        for cnn in self.cnns:
            z = cnn(z)
            z = self.max_pool(z)
        return z

class CNN_Model(keras.models.Model):
    def __init__(self, cnn_layers=5, filters = 64, kernel = 3, fc_neurons = 60, output_dim = 10, **kwargs):
        super().__init__(**kwargs)
        self.cnn_block = CNN_Block(cnn_layers=cnn_layers, filters = filters, kernel = kernel)
        self.fc = keras.layers.Dense(fc_neurons, activation="relu", kernel_initializer="he_normal")
        self.out_layer = keras.layers.Dense(output_dim, activation = "softmax")

    def call(self, inputs):
        z = inputs
        z = self.cnn_block(z)
        z = keras.layers.Flatten()(z)
        z = self.fc(z)
        self.out = self.out_layer(z)

        return self.out


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1,28,28,1)/255.
x_test = x_test.reshape(-1,28,28,1)/255.

# using 15000 samples for check
x_train = x_train[:15000]
y_train = y_train[:15000]


model = CNN_Model()

# since there is only one domain mnist that we are training on, 
# therefore we will find the si's and mi's for all the training examples at once
with tf.GradientTape() as tape2 :
    with tf.GradientTape() as tape1 :
        y_pred = model(x_train, training=True)
        y_loss = keras.losses.SparseCategoricalCrossentropy()(y_train,y_pred)
    df_loss = tape1.gradient(y_loss, model.trainable_variables[-2])

df2_loss = tape2.gradient(df_loss, model.trainable_variables[-2])

s_i = -df_loss*model.trainable_variables[-2]+1/2*df2_loss*model.trainable_variables[-2]*model.trainable_variables[-2]

def impact(x):
    return tf.where(x>0,1.,0.)

# these m_i determine the dropout
m_i = tf.map_fn(impact,s_i)
model.trainable_variables[-2].assign(model.trainable_variables[-2]*m_i)

#custom training routine in order to create the custom dropout gradient

def random_batch (X , y , batch_size = 32): 
    idx = np . random . randint ( len ( X ), size = batch_size ) 
    return X [idx], y [idx] 

def print_status_bar ( iteration , total , loss , metrics = None ): 
    metrics = " " . join ([ "{}: {:.4f}" . format ( m . name , m . result ()) for m in [ loss ] + ( metrics or [])]) 
    end = "" if iteration < total else " \n " 
    print ( " \r {}/{} " . format ( iteration , total ) + metrics , end=end)

n_epochs=3
batch_size=32

n_steps = len(x_train)//batch_size
optimizer = keras.optimizers.Nadam(learning_rate = 0.01) 
loss_fn = keras.losses.SparseCategoricalCrossentropy()
mean_loss = keras.metrics.Mean() 
metrics = [keras.metrics.SparseCategoricalAccuracy()]

for epoch in range (1 , n_epochs + 1): 
    print ( "Epoch {}/{}" . format ( epoch , n_epochs )) 
    for step in range (1 , n_steps + 1 ):
        X_batch , y_batch = random_batch (x_train ,y_train) 
        
        with tf . GradientTape () as tape : 
            y_pred = model (X_batch, training = True ) 
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred )) 
            loss = tf.add_n([ main_loss ] + model.losses) 
        gradients = tape.gradient (loss , model.trainable_variables)
        # the dropped out neurons of ff are not trained
        # gradients[-2] = gradients[-2]*m_i      

        optimizer.apply_gradients(zip(gradients , model.trainable_variables))
        mean_loss(loss) 

        for metric in metrics : 
            metric(y_batch,y_pred) 
            print_status_bar( step * batch_size , len (y_train), mean_loss , metrics) 
            for metric in [mean_loss] + metrics : 
                metric.reset_states() 
    print(" var_acc:",keras.metrics.SparseCategoricalAccuracy()(y_test,model.predict(x_test)).numpy())

model.save_weights("without_dropout.hd5")





    