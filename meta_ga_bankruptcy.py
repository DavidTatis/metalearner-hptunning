# %% [markdown]
# Libreries

# %%

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

import matplotlib.pyplot as plt # plotting library
# %matplotlib inline


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,BatchNormalization,Input
from tensorflow.keras.optimizers import Adam ,RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras import  backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

from sklearn.model_selection import GridSearchCV,ParameterGrid, ParameterSampler,train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error,accuracy_score, balanced_accuracy_score

import random
from random import random,randrange
from operator import itemgetter
import timeit
import random
import os
import pathlib
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="0"
initializer = tf.keras.initializers.GlorotUniform(seed=2)
random.seed(2)

# %%
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/models/')
from fcunet import fcunet_model
from irnet import irnet_model
from fcmnr import fcmnr_model

from Metalearner import meta_learner

# %% [markdown]
# Dataset

# %%
bankruptcy_df = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/data/bankruptcy_dataset.csv")
X = bankruptcy_df.iloc[:,1:]
Y = bankruptcy_df.iloc[:,[0]]
Y=to_categorical(Y)
scaler = RobustScaler()
X.iloc[:,:] = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size=0.3, random_state=42, stratify=Y)

# %% [markdown]
# ### Genetic Algorithm Functions

# %%
#INITIALIZE POPULATION FOR 4 HPs
def initialize_population(use_metalearner,population_size,n_layers,learning_rate,batch_size,activation_function):
  if use_metalearner:
    param_grid = dict(n_layers=top_layers,learning_rate=top_lr,
                  batch_size=top_bs,activation_function=top_af)
  else:
    param_grid = dict(n_layers=n_layers,learning_rate=learning_rate,
                      batch_size=batch_size,activation_function=activation_function)
  grid_search_population=list(ParameterSampler(param_grid,population_size))

  potential_n_layers=[]
  potential_learning_rate=[]
  potential_batch_size=[]
  potential_activation_function=[]

  for i in range(0,population_size):
    potential_n_layers.insert(0,grid_search_population[i]['n_layers'])
    potential_learning_rate.insert(0,grid_search_population[i]['learning_rate'])
    potential_batch_size.insert(0,grid_search_population[i]['batch_size'])
    potential_activation_function.insert(0,grid_search_population[i]['activation_function'])

  return potential_n_layers,potential_learning_rate,potential_batch_size,potential_activation_function

# %%
#EVALUATE FITNESS
def evaluate_fitness(input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,max_epochs,patience_epochs,metric_to_evaluate):
  #CREATE MODEL
  if(selected_arch=="irnet"):
    model=irnet_model(input_shape,n_layers,activation_function,learning_rate) 
  if(selected_arch=="fcunet"):
    model=fcunet_model(input_shape,n_layers,activation_function,learning_rate) 
  if(selected_arch=="fcmnr"):
    model=fcmnr_model(input_shape,n_layers,activation_function,learning_rate) 
  
  start_time = timeit.default_timer()
  history = model.fit(x_train,y_train,
                      batch_size=batch_size,
                      epochs=max_epochs,
                      validation_split=0.2,
                      callbacks=[EarlyStopping(patience=patience_epochs)])
  end_time = timeit.default_timer()

  #EVALUATE MODEL
  prediction=model.predict(x_test)
  if(len(prediction.transpose())!=len(prediction)): # IF RESULT IS ONE-HOT ENCODED, CHANGE IT.
    prediction=np.argmax(prediction,axis=1)

  metric_test=0
  if(metric_to_evaluate=='mae'): metric_test=mean_absolute_error(np.argmax(y_test,axis=1),prediction)
  if(metric_to_evaluate=='accuracy'): metric_test=accuracy_score(np.argmax(y_test,axis=1),prediction)
  if(metric_to_evaluate=='balanced_accuracy'): metric_test=balanced_accuracy_score(np.argmax(y_test,axis=1),prediction)
  
  #SAVE THE WEIGHTS
  weights_name="{}-{}-{}-{}".format(n_layers,input_shape,activation_function,learning_rate)
  model.save(os.path.dirname(os.path.abspath(__file__))+"/data/weights/"+weights_name+".h5")

  #SAVE THE HYPERPARAMS AND THE METRIC
  with open(hp_dataset_name, mode='a+') as hp_dataset:
      hp_dataset_writer=csv.writer(hp_dataset,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      hp_dataset_writer.writerow([selected_arch,
                                  error_metric,
                                  dnn_task,
                                  num_features,
                                  training_samples,
                                  n_layers,
                                  input_shape,
                                  activation_function,
                                  learning_rate,
                                  batch_size,
                                  str(len(history.history['loss'])),
                                  end_time-start_time,
                                  metric_test])
  return metric_test


# %%
def selection(evaluated_hparams,sel_prt,rand_prt,population, metric_to_evaluate,sort_order_desc):
    
    sorted_evaluated_params=sorted(evaluated_hparams,key=itemgetter('metric'),reverse=True)
    if(sel_prt+rand_prt>=len(population[0])):
      print("WARNING: Selections are bigger thant current population")
      print("WARNING: Random selection may not be taken")

    top_selection=[]
    for i in range(sel_prt):
      top_selection.insert(len(top_selection),sorted_evaluated_params[i]['hparam'])

    rand_selection=[]
    i=0
    while(i < rand_prt):
      if(len(rand_selection)+len(top_selection)>=len(population[0])):
        break

      rand_hparam=randrange(len(population[0]))
      print("Generated random {}.".format(rand_hparam))
      if(rand_hparam in top_selection or rand_hparam in rand_selection):
        continue

      rand_selection.insert(0,rand_hparam)
      i=i+1
    return top_selection,rand_selection


# %%
#CROSS-OVER OPERATION

def crossover(p1,p2,population):
    child_potential_n_layers=[]
    child_potential_learning_rate=[]
    child_potential_batch_size=[]
    child_potential_activation_function=[]

    #child1
    child_potential_n_layers.insert(0,population[0][p1])
    child_potential_learning_rate.insert(0,population[1][p2])
    child_potential_batch_size.insert(0,population[2][p1])
    child_potential_activation_function.insert(0,population[3][p2])

    #child2
    child_potential_n_layers.insert(0,population[0][p2])
    child_potential_learning_rate.insert(0,population[1][p1])
    child_potential_batch_size.insert(0,population[2][p2])
    child_potential_activation_function.insert(0,population[3][p1])
    
    #child3
    child_potential_n_layers.insert(0,population[0][p1])
    child_potential_learning_rate.insert(0,population[1][p1])
    child_potential_batch_size.insert(0,population[2][p2])
    child_potential_activation_function.insert(0,population[3][p2])
    
    #child4
    child_potential_n_layers.insert(0,population[0][p2])
    child_potential_learning_rate.insert(0,population[1][p2])
    child_potential_batch_size.insert(0,population[2][p1])
    child_potential_activation_function.insert(0,population[3][p1])
    
    
    child_hparams=[child_potential_n_layers,child_potential_learning_rate,child_potential_batch_size,child_potential_activation_function]
    return child_hparams


# %%

# MUTATION
def mutation(population,selected):
    selected_hyperparam=randrange(len(all_hyperparams))
    selected_value=randrange(len(all_hyperparams[selected_hyperparam]))
    population[selected_hyperparam][selected]=all_hyperparams[selected_hyperparam][selected_value]
    

# %%
# MAIN
def genetic_algorithm_main(use_metalearner,population_size,input_shape,hp_dataset_name,max_epochs,patience_epochs,metric_to_evaluate,sort_order_desc):
   
    potential_n_layers,potential_learning_rate,potential_batch_size,potential_activation_function=initialize_population(use_metalearner,
                                                                                                population_size=population_size,
                                                                                                n_layers=n_layers,
                                                                                                learning_rate=learning_rate,
                                                                                                batch_size=batch_size,
                                                                                                activation_function=activation_function)
    
    population=[potential_n_layers,potential_learning_rate,potential_batch_size,potential_activation_function]
    print("Initial population",population)
    final_hyperparam=[]
    # evaluate hyperparams
    for generation in range(generations):
        evaluated_hparams=[]
        for i in range(population_size):
            #input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,max_epochs,patience_epochs,metric_to_evaluate
            metric=evaluate_fitness(input_shape=input_shape,
                                    n_layers=population[0][i],
                                    learning_rate=population[1][i],
                                    batch_size=population[2][i],
                                    activation_function=population[3][i],
                                    hp_dataset_name=hp_dataset_name,
                                    max_epochs=max_epochs,
                                    patience_epochs=patience_epochs,
                                    metric_to_evaluate=metric_to_evaluate)
            evaluated_hparams.insert(0,{"hparam":i,"metric":metric})

        #SELECTION
        top_selection,rand_selection=selection(evaluated_hparams,sel_prt,rand_prt,population,metric_to_evaluate,sort_order_desc)


        # CROSS-OVER
        p1,p2=random.sample(range(0,len(top_selection)+len(rand_selection)),2)
        child_hyperparams= crossover(p1,p2,population)

        #CREATE NEW POPULATION
        #insert top selections
        new_population=[[population[0][i] for i in top_selection],
                          [population[1][i] for i in top_selection],
                          [population[2][i] for i in top_selection],
                          [population[3][i] for i in top_selection]]
        #insert random selection and childs
        new_population[0]=[*new_population[0],
                                    *[population[0][i] for i in rand_selection],
                                   *child_hyperparams[0]]
        new_population[1]=[*new_population[1],
                                    *[population[1][i] for i in rand_selection],
                                   *child_hyperparams[1]]
        new_population[2]=[*new_population[2],
                                    *[population[2][i] for i in rand_selection],
                                   *child_hyperparams[2]]
        new_population[3]=[*new_population[3],
                                    *[population[3][i] for i in rand_selection],
                                   *child_hyperparams[3]]    

        # MUTATION
        selected_to_mutate=randrange(len(top_selection)+len(rand_selection)+len(child_hyperparams[0]))
        mutation(new_population,selected_to_mutate)

        if (generation+1)==generations:
            for  hyperparam in  population:
                final_hyperparam.insert(len(population),hyperparam[top_selection[0]])

        population=new_population
        population_size=len(population[0])

    return evaluated_hparams,sorted(evaluated_hparams,key=itemgetter('metric'),reverse=sort_order_desc)[0]['metric'],final_hyperparam
    
                


# %%


#FILES NAME
hp_dataset_name="test_hp_dataset.csv"
weights_folder="data/weights/"
#HYPERPARAMETERS TO EVALUATE
num_features=x_train.shape[1]
training_samples=x_train.shape[0]
training_and_validation_samples=len(x_train)
n_layers=[1,2,3]
learning_rate=[0.01,0.001,0.0001,0.00001]
batch_size=[16,32,64,128]
activation_function=['relu','elu','tanh','sigmoid']


# METALEARNER
to_categorical_column_names=["activation_function"] 
n_top_hp_to_select=2
dataset_column_names=["architecture","error_metric","task","num_features",
                    "training_samples","n_layers","activation_function",
                    "learning_rate","batch_size","metric","dimension","dataset","y"]

x_column_names=["num_features","training_samples",
                    "n_layers","activation_function",
                    "learning_rate", "batch_size"]

to_categorical_column_names=["activation_function"]
data_file_name="data/metadataset.csv"
dnn_architecture="all"
dnn_task="prediction"
dnn_dim=1



top_lr,top_bs,top_layers,top_af,finish_order,selected_arch=meta_learner(dnn_architecture,dnn_task,dnn_dim,n_top_hp_to_select,dataset_column_names,x_column_names,
                to_categorical_column_names,data_file_name,num_features,training_samples,n_layers,learning_rate,batch_size,activation_function)
print(selected_arch,finish_order)


input_shape=x_train.shape[1]
max_epochs=2
patience_epochs=2
metric_to_evaluate="mae"
error_metric=1
sort_order_desc=True



#GA configuration
all_hyperparams=[n_layers,learning_rate,batch_size,activation_function]
population_size=4
sel_prt=2
rand_prt=1
generations=3

use_metalearner=True

all_ga,top_ga, hparams_ga=genetic_algorithm_main(use_metalearner,
                                                population_size,
                                                input_shape,
                                                hp_dataset_name,
                                                max_epochs,
                                                patience_epochs,
                                                metric_to_evaluate,
                                                sort_order_desc)

# %%



