
# %%
import numpy as np # linear algebra
from numpy import genfromtxt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

import matplotlib.pyplot as plt # plotting library

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,BatchNormalization,Input
from tensorflow.keras.optimizers import Adam ,RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras import  backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image,ImageOps
from tensorflow import keras
from tensorflow.keras import layers

# from tensorflow.keras.data import Dataset
from sklearn.model_selection import GridSearchCV,ParameterGrid, ParameterSampler,train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error,accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
import random
from random import random,randrange
from operator import itemgetter
import timeit
import random
import os
import pathlib
import pickle
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"]="0"
initializer = tf.keras.initializers.GlorotUniform(seed=2)
random.seed(22)

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/models/')
from Unet import segmentation_unet_model,unet_model
from convIrnet import conv_irnet_model
from MnR import mnr_model
from cnn import cnn_model
import scipy.io
from spectral import open_image,imshow
from Metalearner import meta_learner

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# %%

data = scipy.io.loadmat(os.path.dirname(os.path.abspath(__file__))+"/data/enrique/HS1m.mat")
x = data['HS1m']
# data2 = scipy.io.loadmat(os.path.dirname(os.path.abspath(__file__))+"/data/enrique/gt_enrique_1m_cut.mat")
data2 = scipy.io.loadmat(os.path.dirname(os.path.abspath(__file__))+"/data/enrique/gt_corrected.mat")
gt = data2['pix_gt']

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

num_pca=30
patch_size=4
x_pca,pca=applyPCA(x,numComponents=num_pca)
gt=np.reshape(gt,(gt.shape[0],gt.shape[1],1))

#RESHAPE THE DATA TO HAVE EACH PIXEL IN A LIST (TOTAL_NUMBER_OF_PIXELS, NUM_OF_PCA)
data2=np.reshape(x_pca,[x.shape[0]*x.shape[1],x_pca.shape[2]])
gt2=gt.ravel()
#REMOVE PIXELS OF CLASS 0
data2=data2[gt2>0,:]
gt2=gt2[gt2>0]
print(data2.shape)
print(gt2.shape)

gt2 -=1
print(data2.shape,gt2.shape)


#AS PIXELS WERE DELETED, TO CREATE EXACT PATCHES WE NEED TO REMOVE RESIDUALS PIXELS
extra_pixels_to_delete=gt2.shape[0]%(patch_size*patch_size)
print(extra_pixels_to_delete)
#RESHAPE ALL THE PIXEL TO PATCHES, SAME TO THE GT
gt_patches=np.reshape(gt2[:-extra_pixels_to_delete],(int(gt2.shape[0]/(patch_size*patch_size)),patch_size,patch_size))
data_patches=np.reshape(data2[:-extra_pixels_to_delete,],(int(data2.shape[0]/(patch_size*patch_size)),patch_size,patch_size,num_pca))

#CHANGE THE GT TO CATEGORICAL
gt_patches_cat=to_categorical(gt_patches, num_classes = 6)

# %%
# view = imshow(data_patches[0],(1,15,20),classes=gt_patches[0])
# view.set_display_mode('overlay')
# view.class_alpha = 0
# view = imshow(data_patches[1],(1,50,100),classes=gt_patches[1])
# view.set_display_mode('overlay')
# view.class_alpha = 0
xtrain,xtest,ytrain,ytest = train_test_split(data_patches,gt_patches_cat,random_state=42, test_size=0.25)


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
def evaluate_fitness(input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,max_epochs,patience_epochs,metric_to_evaluate,total_population):
  #CHECK IF THE HPS HAVE BEEN USED SO RETURN THE METRIC WITHOUT TRAINING AGAIN
  for j in range(len(total_population[0])):
    if (n_layers==total_population[0][j] and
        learning_rate==total_population[1][j] and
        batch_size==total_population[2][j] and
        activation_function==total_population[3][j]):
        return total_population[4][j]

  #CREATE MODEL
  if(selected_arch=="CNN"):
    model=cnn_model(input_shape=(patch_size,patch_size,num_pca),n_layers=n_layers,activation_function=activation_function,learning_rate=learning_rate)

  earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min', 
                              verbose=1, 
                              patience=patience_epochs
                              )
  weights_name="{}-{}-{}-{}".format(n_layers,batch_size,activation_function,learning_rate)
  callbacks = [earlystopping]
  
  start_time = timeit.default_timer()
  
  history = model.fit(x=xtrain,y=ytrain, epochs=max_epochs,
                  callbacks=callbacks,
                  batch_size=batch_size,
                  validation_split=0.2)
  end_time = timeit.default_timer()

  results = model.evaluate(xtest,ytest)
  print("Test IOU, Acc: ",results[2],results[1])

  metric_test=results[1]
  #SAVE THE WEIGHTS
  
  # model.save(os.path.dirname(os.path.abspath(__file__))+"/data/weights/"+weights_name+".h5")

  #SAVE THE HYPERPARAMS AND THE METRIC
  with open(os.path.dirname(os.path.abspath(__file__))+"/data/"+hp_dataset_name, mode='a+') as hp_dataset:
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
  
  total_population[0].insert(len(total_population[0]),n_layers)
  total_population[1].insert(len(total_population[1]),learning_rate)
  total_population[2].insert(len(total_population[2]),batch_size)
  total_population[3].insert(len(total_population[3]),activation_function)
  total_population[4].insert(len(total_population[4]),metric_test)
  
  return metric_test


# %%
def selection(evaluated_hparams,sel_prt,rand_prt,population):
    if error_metric==1:
      order_desc=False
    else:
      order_desc=True
    sorted_evaluated_params=sorted(evaluated_hparams,key=itemgetter('metric'),reverse=order_desc)
    if(sel_prt+rand_prt>=len(population[0])):
      print("WARNING: Selections are bigger thant current population")
      print("WARNING: Random selection may not be taken")
    if len(population[0])<2:
      raise Exception("POPULATION < 2")
    top_selection=[]
    for i in range(sel_prt):
      top_selection.insert(len(top_selection),sorted_evaluated_params[i]['hparam'])

    rand_selection=[]
    if len(population[0])==2:
      print("WARNING: no random selection because population len == 2")
      return top_selection,rand_selection
    
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
def mutation(new_population,selected):
    selected_hyperparam=randrange(len(all_hyperparams))
    selected_value=randrange(len(all_hyperparams[selected_hyperparam]))
    new_population[selected_hyperparam][selected]=all_hyperparams[selected_hyperparam][selected_value]
    return new_population
    

# %%
# MAIN
def genetic_algorithm_main(use_metalearner,population_size,input_shape,hp_dataset_name,max_epochs,patience_epochs,metric_to_evaluate):
   
    potential_n_layers,potential_learning_rate,potential_batch_size,potential_activation_function=initialize_population(use_metalearner,
                                                                                                population_size=population_size,
                                                                                                n_layers=n_layers,
                                                                                                learning_rate=learning_rate,
                                                                                                batch_size=batch_size,
                                                                                                activation_function=activation_function)
    
    population=[potential_n_layers,potential_learning_rate,potential_batch_size,potential_activation_function]
    total_population=[[],[],[],[],[]] # n_laters, learning_rate, batch_size, activation_function, metric
    final_hyperparam=[]
    # evaluate hyperparams
    for generation in range(generations):
        evaluated_hparams=[]
        for i in range(population_size):
            #input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,max_epochs,patience_epochs,metric_to_evaluate
            metric=evaluate_fitness(input_shape=input_shape,
                                    n_layers=int(population[0][i]),
                                    learning_rate=float(population[1][i]),
                                    batch_size=int(population[2][i]),
                                    activation_function=population[3][i],
                                    hp_dataset_name=hp_dataset_name,
                                    max_epochs=max_epochs,
                                    patience_epochs=patience_epochs,
                                    metric_to_evaluate=metric_to_evaluate,
                                    total_population=total_population)
            evaluated_hparams.insert(0,{"hparam":i,"metric":metric})

        #SELECTION
        top_selection,rand_selection=selection(evaluated_hparams,sel_prt,rand_prt,population)


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
        new_population=mutation(new_population,selected_to_mutate)

        
        if (generation+1)==generations:
            for  hyperparam in  population:
                final_hyperparam.insert(len(population),hyperparam[top_selection[0]])

        population=new_population
        population_size=len(population[0])
    
    if error_metric==True:
      order_desc=False
    else:
      order_desc=True
    return evaluated_hparams,sorted(evaluated_hparams,key=itemgetter('metric'),reverse=order_desc)[0]['metric'],final_hyperparam
    
                


# %%


#FILES NAME
hp_dataset_name="metadata_enrique_no-ml.csv"
weights_folder="data/weights/"
data_file_name="data/metadataset.csv"

#HYPERPARAMETERS TO EVALUATE
n_layers = [1,2,3,4,5]
activation_function=['relu','tanh','sigmoid','elu']
learning_rate=[0.01,0.001,0.0001,0.00001]
batch_size=[8,16,32,64,128]
max_epochs=1000
patience_epochs=200

# METALEARNER
num_features=30
training_samples=len(xtrain)
to_categorical_column_names=["activation_function"] 
dataset_column_names=["architecture","error_metric","task","num_features",
                    "training_samples","n_layers","activation_function",
                    "learning_rate","batch_size","metric","dimension","dataset","y"]
x_column_names=["num_features","training_samples",
                    "n_layers","activation_function",
                    "learning_rate", "batch_size"]
to_categorical_column_names=["activation_function"]

n_top_hp_to_select=2
dnn_architecture="CNN"
dnn_task="segmentation"
dnn_dim=3



top_lr,top_bs,top_layers,top_af,finish_order,selected_arch=meta_learner(dnn_architecture,dnn_task,dnn_dim,n_top_hp_to_select,dataset_column_names,x_column_names,
                to_categorical_column_names,data_file_name,num_features,training_samples,n_layers,learning_rate,batch_size,activation_function)
print(selected_arch)


input_shape=(patch_size,patch_size,num_pca)
metric_to_evaluate="acc"
error_metric=0




#GA configuration
all_hyperparams=[n_layers,learning_rate,batch_size,activation_function]
population_size=4
sel_prt=2
rand_prt=1
generations=3

use_metalearner=False

all_ga,top_ga, hparams_ga=genetic_algorithm_main(use_metalearner,
                                                population_size,
                                                input_shape,
                                                hp_dataset_name,
                                                max_epochs,
                                                patience_epochs,
                                                metric_to_evaluate)

# %%



