from distutils.log import error
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, BatchNormalization,Input,concatenate
from tensorflow.keras.models import Model

#TODO: IMPLEMENT THE N LAYERS

def fcunet_model(n_layers,input_shape,activation_function,learning_rate):   
    if n_layers==1:
        return fcunet_model_1(input_shape,activation_function,learning_rate)
    if n_layers==2:
      return fcunet_model_2(input_shape,activation_function,learning_rate)
    if n_layers == 3:
      return fcunet_model_3(input_shape,activation_function,learning_rate)
    
    


def fcunet_model_1(input_shape,activation_function,learning_rate):
    input_vec = Input(shape=input_shape)
    
    #encoder
    x3=Activation(activation_function)(BatchNormalization()(Dense(1024)(input_vec)))
    x6=Activation(activation_function)(BatchNormalization()(Dense(512)(x3)))
    x9=Activation(activation_function)(BatchNormalization()(Dense(256)(x6)))
    x12=Activation(activation_function)(BatchNormalization()(Dense(128)(x9)))
    x15=Activation(activation_function)(BatchNormalization()(Dense(64)(x12)))

    #decoder
    merge1=concatenate([x12, x15], axis=-1)
    x18=Activation(activation_function)(BatchNormalization()(Dense(128)(merge1)))

    merge2=concatenate([x9, x18], axis=-1)
    x21=Activation(activation_function)(BatchNormalization()(Dense(256)(merge2)))

    merge3=concatenate([x6, x21], axis=-1)
    x24=Activation(activation_function)(BatchNormalization()(Dense(128)(merge3)))

    merge4=concatenate([x3, x24], axis=-1)
    x27=Activation(activation_function)(BatchNormalization()(Dense(16)(merge4)))

    x28=Dense(1, activation='linear')(x27) #53

    model=Model(input_vec,x28)
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=tf.keras.losses.mean_absolute_error,
      metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    return model

def fcunet_model_2(input_shape,activation_function,learning_rate):
    input_vec = Input(shape=input_shape)
    #encoder
    x0=Activation(activation_function)(BatchNormalization()(Dense(1024)(input_vec)))
    x3=Activation(activation_function)(BatchNormalization()(Dense(1024)(x0)))

    x4=Activation(activation_function)(BatchNormalization()(Dense(512)(x3)))
    x6=Activation(activation_function)(BatchNormalization()(Dense(512)(x4)))

    x7=Activation(activation_function)(BatchNormalization()(Dense(256)(x6)))
    x9=Activation(activation_function)(BatchNormalization()(Dense(256)(x7)))

    x10=Activation(activation_function)(BatchNormalization()(Dense(128)(x9)))
    x12=Activation(activation_function)(BatchNormalization()(Dense(128)(x10)))

    x13=Activation(activation_function)(BatchNormalization()(Dense(64)(x12)))
    x15=Activation(activation_function)(BatchNormalization()(Dense(64)(x13)))

    #decoder
    merge1=concatenate([x12, x15], axis=-1)
    x16=Activation(activation_function)(BatchNormalization()(Dense(128)(merge1)))
    x18=Activation(activation_function)(BatchNormalization()(Dense(128)(x16)))

    merge2=concatenate([x9, x18], axis=-1)
    x19=Activation(activation_function)(BatchNormalization()(Dense(256)(merge2)))
    x21=Activation(activation_function)(BatchNormalization()(Dense(256)(x19)))

    merge3=concatenate([x6, x21], axis=-1)
    x22=Activation(activation_function)(BatchNormalization()(Dense(256)(merge3)))
    x24=Activation(activation_function)(BatchNormalization()(Dense(128)(x22)))

    merge4=concatenate([x3, x24], axis=-1)
    x25=Activation(activation_function)(BatchNormalization()(Dense(64)(merge4)))
    x27=Activation(activation_function)(BatchNormalization()(Dense(16)(x25)))

    x28=Dense(1, activation='linear')(x27)

    model=Model(input_vec,x28)
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=tf.keras.losses.mean_absolute_error,
      metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

def fcunet_model_3(input_shape,activation_function,learning_rate):
    input_vec = Input(shape=input_shape)
    #encoder
    x0=Activation(activation_function)(BatchNormalization()(Dense(1024)(input_vec)))
    x1=Activation(activation_function)(BatchNormalization()(Dense(1024)(x0)))
    x2=Activation(activation_function)(BatchNormalization()(Dense(1024)(x1)))
    x3=Activation(activation_function)(BatchNormalization()(Dense(1024)(x2)))

    x4=Activation(activation_function)(BatchNormalization()(Dense(512)(x3)))
    x5=Activation(activation_function)(BatchNormalization()(Dense(512)(x4)))
    x6=Activation(activation_function)(BatchNormalization()(Dense(512)(x5)))

    x7=Activation(activation_function)(BatchNormalization()(Dense(256)(x6)))
    x8=Activation(activation_function)(BatchNormalization()(Dense(256)(x7)))
    x9=Activation(activation_function)(BatchNormalization()(Dense(256)(x8)))

    x10=Activation(activation_function)(BatchNormalization()(Dense(128)(x9)))
    x11=Activation(activation_function)(BatchNormalization()(Dense(128)(x10)))
    x12=Activation(activation_function)(BatchNormalization()(Dense(128)(x11)))

    x13=Activation(activation_function)(BatchNormalization()(Dense(64)(x12)))
    x14=Activation(activation_function)(BatchNormalization()(Dense(64)(x13)))
    x15=Activation(activation_function)(BatchNormalization()(Dense(64)(x14)))

    #decoder
    merge1=concatenate([x12, x15], axis=-1)
    x16=Activation(activation_function)(BatchNormalization()(Dense(128)(merge1)))
    x17=Activation(activation_function)(BatchNormalization()(Dense(128)(x16)))
    x18=Activation(activation_function)(BatchNormalization()(Dense(128)(x17)))

    merge2=concatenate([x9, x18], axis=-1)
    x19=Activation(activation_function)(BatchNormalization()(Dense(256)(merge2)))
    x20=Activation(activation_function)(BatchNormalization()(Dense(256)(x19)))
    x21=Activation(activation_function)(BatchNormalization()(Dense(256)(x20)))

    merge3=concatenate([x6, x21], axis=-1)
    x22=Activation(activation_function)(BatchNormalization()(Dense(256)(merge3)))
    x23=Activation(activation_function)(BatchNormalization()(Dense(256)(x22)))
    x24=Activation(activation_function)(BatchNormalization()(Dense(128)(x23)))

    merge4=concatenate([x3, x24], axis=-1)
    x25=Activation(activation_function)(BatchNormalization()(Dense(64)(merge4)))
    x26=Activation(activation_function)(BatchNormalization()(Dense(32)(x25)))
    x27=Activation(activation_function)(BatchNormalization()(Dense(16)(x26)))

    x28=Dense(1, activation='linear')(x27) #53

    model=Model(input_vec,x28)
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=tf.keras.losses.mean_absolute_error,
      metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    return model