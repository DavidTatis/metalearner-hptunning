import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model




def FCMnR_model(n_layers,input_shape,activation_function,learning_rate):
    input_vec = Input(shape=input_shape)
    x0=Activation(activation_function)(BatchNormalization()((Dense(1024)(input_vec))))
    x1=Activation(activation_function)(BatchNormalization()((Dense(1024)(x0))))
    x2=Activation(activation_function)(Dense(1)(x1))
    
    x1_3=Activation(activation_function)(BatchNormalization()((Dense(512)(x2))))
    x2_3=Activation(activation_function)(BatchNormalization()((Dense(512)(x2))))
    
    x1_4=Activation(activation_function)(Dense(128)(x1_3))
    x2_4=Activation(activation_function)(Dense(128)(x2_3))
    
    x1_5=Dense(1, activation=activation_function)(x1_4)
    x2_5=Dense(1, activation=activation_function)(x2_4)
    
    sum0=x2+x2
    sum1_1=x1_5+sum0
    sum2_1=x2_5+sum0
    
    #CHANGE THE "NUMBER OF LAYERS" TO A NUMBER OF MNR BLOCKS
    if(n_layers==1):
        n=2
    if(n_layers==2):
        n=3
    if(n_layers==3):
        n=4
        
    for i in range(n):
        avg1=(sum1_1+sum2_1)/2.
        
        MnRCSNet_1_1=Activation(activation_function)(BatchNormalization()(Dense(1024)(sum1_1)))
        MnRCSNet_2_1=Activation(activation_function)(BatchNormalization()(Dense(1024)(sum2_1)))
        
        MnRCSNet_1_2=Activation(activation_function)(BatchNormalization()(Dense(512)(MnRCSNet_1_1)))
        MnRCSNet_2_2=Activation(activation_function)(BatchNormalization()(Dense(512)(MnRCSNet_2_1)))
        
        MnRCSNet_1_3=Activation(activation_function)(Dense(1)(MnRCSNet_1_2))
        MnRCSNet_2_3=Activation(activation_function)(Dense(1)(MnRCSNet_2_2))
        
        sum1_1=MnRCSNet_1_3+avg1
        sum2_1=MnRCSNet_2_3+avg1
    
    
    x1_6=Activation(activation_function)(BatchNormalization()(Dense(512)(sum1_1)))
    x2_6=Activation(activation_function)(BatchNormalization()(Dense(512)(sum2_1)))

    x1_7=Activation(activation_function)(BatchNormalization()(Dense(256)(x1_6)))
    x2_7=Activation(activation_function)(BatchNormalization()(Dense(256)(x2_6)))

    x1_8=Activation(activation_function)(Dense(1)(x1_7))
    x2_8=Activation(activation_function)(Dense(1)(x2_7))
    avg2=(sum1_1+sum2_1)/2.
    sum_final=x1_8+x2_8+avg2
    x_final=Activation('linear')(Dense(1)(sum_final))
     
    model=Model(input_vec,x_final)
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=tf.keras.losses.mean_absolute_error,
      metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    return model
