import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.layers import Input,concatenate
from tensorflow.keras.models import Model



def irnet_model(input_shape,n_layers,activation_function,learning_rate):   
    input_vec = Input(shape=input_shape)
    x0=Activation(activation_function)(BatchNormalization()(Dense(1024)(input_vec)))
    m1=concatenate([input_vec, x0], axis=-1)
    
    for i in range(n_layers):
        x1=Activation(activation_function)(BatchNormalization()(Dense(1024)(m1)))
        m1=concatenate([m1, x1], axis=-1)
    for i in range(n_layers):
        x1=Activation(activation_function)(BatchNormalization()(Dense(512)(m1)))
        m1=concatenate([m1, x1], axis=-1)

    for i in range(n_layers):
        x1=Activation(activation_function)(BatchNormalization()(Dense(256)(m1)))
        m1=concatenate([m1, x1], axis=-1)

    for i in range(n_layers):
        x1=Activation(activation_function)(BatchNormalization()(Dense(128)(m1)))
        m1=concatenate([m1, x1], axis=-1)
        
    for i in range(n_layers):
        x1=Activation(activation_function)(BatchNormalization()(Dense(64)(m1)))
        m1=concatenate([m1, x1], axis=-1)

    for i in range(n_layers):
        x1=Activation(activation_function)(BatchNormalization()(Dense(32)(m1)))
        m1=concatenate([m1, x1], axis=-1)
    for i in range(n_layers):
        x1=Activation(activation_function)(BatchNormalization()(Dense(16)(m1)))
        m1=concatenate([m1, x1], axis=-1)
            
    xf=Dense(2,activation='softmax')(m1)

    model=Model(input_vec,xf)
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=tf.keras.losses.binary_crossentropy,
      metrics=['accuracy'])
    
    return model