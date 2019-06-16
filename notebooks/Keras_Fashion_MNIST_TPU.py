#!/usr/bin/env python
# coding: utf-8

# ## Keras hyperparameter optimization - TPUs
# [How to perform Keras hyperparameter optimization x3 faster on TPU for free](https://www.dlology.com/blog/how-to-perform-keras-hyperparameter-optimization-on-tpu-for-free/)

# In[1]:


import numpy as np
import keras
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[ ]:


x = np.concatenate((x_train, x_test), axis=0)


# In[3]:


x.shape


# In[4]:


y = np.concatenate((y_train, y_test), axis=0)
y.shape


# In[ ]:


get_ipython().system('pip install -q talos')


# In[ ]:


para = {
    'dense1_neuron': [256, 512],
    'activation': ['relu', 'elu'],
    'conv_dropout': [0.25, 0.4]
}


# In[ ]:


import tensorflow as tf
import os
def fashion_mnist_fn_tpu(x_train, y_train, x_val, y_val, params):
    tf.keras.backend.clear_session()
    conv_dropout = float(params['conv_dropout'])
    dense1_neuron = int(params['dense1_neuron'])
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation=params['activation']))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(conv_dropout))

    model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation=params['activation']))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(conv_dropout))

    model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation=params['activation']))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(conv_dropout))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params['dense1_neuron']))
    model.add(tf.keras.layers.Activation(params['activation']))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    
    tpu_model = tf.contrib.tpu.keras_to_tpu_model(
        model,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        )
    )
    tpu_model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-3, ),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['categorical_accuracy']
    )


    out = tpu_model.fit(
        x, y, epochs=10, batch_size = 1024,
        verbose=0,
        validation_data=[x_val, y_val]
    )
    
    return out, tpu_model.sync_to_cpu()


# In[ ]:


import talos as ta


# In[10]:


scan_results = ta.Scan(x, y, para, fashion_mnist_fn_tpu)


# In[25]:


scan_results.data


# In[34]:


model_id = scan_results.data['val_categorical_accuracy'].astype('float').argmax() - 1
model_id + 1


# In[ ]:


tf.keras.backend.clear_session()
from tensorflow.keras.models import model_from_json
model = model_from_json(scan_results.saved_models[model_id])
model.set_weights(scan_results.saved_weights[model_id])


# In[36]:


model.summary()


# In[ ]:


model.save('./best_model.h5')


# In[ ]:


from google.colab import files

files.download('./best_model.h5')


# In[13]:


# access the summary details
scan_results.details


# In[19]:


# use Scan object as input
report = ta.Reporting(scan_results)
# access the dataframe with the results
report.data.head(-3)


# In[20]:


# get the number of rounds in the Scan
report.rounds()


# In[21]:


# get the highest result for any metric
report.high('categorical_accuracy')

