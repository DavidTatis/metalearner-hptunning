# %% [markdown]
# Libreries

# %%
import numpy as np # linear algebra
from numpy import genfromtxt
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image,ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import gaussian_filter
# from tensorflow.keras.data import Dataset
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
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"]="0"
initializer = tf.keras.initializers.GlorotUniform(seed=2)
random.seed(22)

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/models/')
from Unet import segmentation_unet_model,unet_model
from convIrnet import conv_irnet_model
from MnR import mnr_model

from Metalearner import meta_learner

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ==========

orgs = glob(os.path.dirname(os.path.abspath(__file__))+"/data/EM/images/*")
masks = glob(os.path.dirname(os.path.abspath(__file__))+"/data/EM/labels/*")
imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image)))
    masks_list.append(np.array(Image.open(mask)))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)
x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)/255
y[y>=0.5]=1
y[y<0.5]=0

def get_patches(img_arr, size=256, stride=256):
    """
    Takes single image or array of images and returns
    crops using sliding window method.
    If stride < size it will do overlapping.
    
    Args:
        img_arr (numpy.ndarray): [description]
        size (int, optional): [description]. Defaults to 256.
        stride (int, optional): [description]. Defaults to 256.
    
    Raises:
        ValueError: [description]
        ValueError: [description]
    
    Returns:
        numpy.ndarray: [description]
    """    
    # check size and stride
    if size % stride != 0:
        raise ValueError("size % stride must be equal 0")

    patches_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1

    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping

        for i in range(i_max):
            for j in range(i_max):
                # print(i*stride, i*stride+size)
                # print(j*stride, j*stride+size)
                patches_list.append(
                    img_arr[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size
                    ]
                )

    elif img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[
                            i * stride : i * stride + size,
                            j * stride : j * stride + size,
                        ]
                    )

    else:
        raise ValueError("img_arr.ndim must be equal 3 or 4")

    return np.stack(patches_list)

x_copy=[]
y_copy=[]
for i in range(len(x)):
  im_merge = np.concatenate((x[i][..., None], y[i][..., None]), axis=2)
  im_merge_t = im_merge
  #im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3,8,8)
  img = im_merge_t[..., 0]
  label = im_merge_t[..., 1]
  #img=input_filled_mirroring(img)
  img = np.expand_dims(img,0) 
  label = np.expand_dims(label,0)
  x_copy.append(img)
  y_copy.append(label)
  
x_copy=np.array(x_copy)
y_copy=np.array(y_copy)

y_array = y_copy.reshape(y_copy.shape[0], y_copy.shape[2], y_copy.shape[3], 1)
x_array = x_copy.reshape(x_copy.shape[0], x_copy.shape[2], x_copy.shape[3], 1)

print("Creating patches")
x_array=get_patches(x_array,size=128,stride=128)
y_array=get_patches(y_array,size=128,stride=128)

print("Creating oneHot")
y_array_cropp=[]
input_size=128
output_size=128
start_pixel=int((input_size-output_size)/2)
end_pixel=start_pixel+output_size
for image in y_array:
  newImage=image[start_pixel:end_pixel,start_pixel:end_pixel,0]#CENTER OF THE OUTPUT IMAGE (RESULT OF THE CONVS IN THE INPUT)
  newImage=np.expand_dims(newImage,axis=2)
  y_array_cropp.append(newImage)
y_array_cropp=np.array(y_array_cropp)
y_array_oneHot=[[[[0.,0.] for y in y_array_cropp[0][0]]for x in y_array_cropp[0]] for n in y_array_cropp]
y_array_oneHot=np.array(y_array_oneHot)

for n in range(0,len(y_array_cropp)):
  for i in range(0,len(y_array_cropp[0])):
    for j in range(0, len(y_array_cropp[0][0])):
      if(y_array_cropp[n][i][j]>=0.5):
        y_array_oneHot[n][i][j][1]=1.0
        y_array_oneHot[n][i][j][0]=0.0
      else:
        y_array_oneHot[n][i][j][1]=0.0
        y_array_oneHot[n][i][j][0]=1.0
print(y_array_oneHot.shape)
print(np.max(x_array),np.max(y_array_oneHot))

#to use one-hot replace y_array_cropp for y_array_oneHot
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array_oneHot, test_size=0.1, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
# =============

print(x_train.shape,y_train.shape)

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
  if(selected_arch=="irnet"):
    model=conv_irnet_model(input_shape,n_layers,activation_function,learning_rate) 
  if(selected_arch=="unet"):
    model=unet_model(input_shape,n_layers,activation_function,learning_rate) 
  if(selected_arch=="mnr"):
    model=mnr_model(input_shape,n_layers,activation_function,learning_rate) 
  
  
# Split our img paths into a training and a validation set
  data_gen_args = dict(rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=50,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        validation_split=0.1)

  image_datagen = ImageDataGenerator(**data_gen_args)
  mask_datagen = ImageDataGenerator(**data_gen_args)
  seed = 0
  image_datagen.fit(x_train, augment=True, seed=seed)
  mask_datagen.fit(y_train, augment=True, seed=seed)
  

  image_generator = image_datagen.flow(
          x_train,
          batch_size=batch_size,
          shuffle=True,
          seed=seed)
  ## set the parameters for the data to come from (masks)
  mask_generator = mask_datagen.flow(
          y_train,
          batch_size=batch_size,
          shuffle=True,
          seed=seed)

  val_image_generator = image_datagen.flow(
          x_val,
          batch_size=batch_size,
          shuffle=True,
          seed=seed)
  ## set the parameters for the data to come from (masks)
  val_mask_generator = mask_datagen.flow(
          y_val,
          batch_size=batch_size,
          shuffle=True,
          seed=seed)  

  # combine generators into one which yields image and masks
  train_generator = (pair for pair in zip(image_generator, mask_generator))
  val_generator = (pair for pair in zip(val_image_generator, val_mask_generator))
  history=model.fit_generator(
      train_generator,
      steps_per_epoch=len(x_train) // batch_size,
      validation_data=val_generator,
      validation_steps=1,
      callbacks=[EarlyStopping(patience=patience_epochs)],
      epochs=max_epochs)
   
  start_time= timeit.default_timer()
  
  end_time = timeit.default_timer()

  
  results = model.evaluate(x_test,y_test)
  print("Test IOU: ",results)
  metric_test=results[1]
  #SAVE THE WEIGHTS
  weights_name="{}-{}-{}-{}".format(n_layers,batch_size,activation_function,learning_rate)
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
hp_dataset_name="metadata_isbi2012_ml.csv"
weights_folder="data/weights/"
data_file_name="data/metadataset.csv"

#HYPERPARAMETERS TO EVALUATE
num_features=3
training_samples=len(x_train)+len(x_val)
n_layers=[1,2,3]
learning_rate=[0.01,0.001,0.0001,0.00001]
batch_size=[8,16,32,64]
activation_function=['relu','elu','tanh','sigmoid']


# METALEARNER
to_categorical_column_names=["activation_function"] 
dataset_column_names=["architecture","error_metric","task","num_features",
                    "training_samples","n_layers","activation_function",
                    "learning_rate","batch_size","metric","dimension","dataset","y"]
x_column_names=["num_features","training_samples",
                    "n_layers","activation_function",
                    "learning_rate", "batch_size"]
to_categorical_column_names=["activation_function"]

n_top_hp_to_select=2
dnn_architecture="all"
dnn_task="segmentation"
dnn_dim=2



top_lr,top_bs,top_layers,top_af,finish_order,selected_arch=meta_learner(dnn_architecture,dnn_task,dnn_dim,n_top_hp_to_select,dataset_column_names,x_column_names,
                to_categorical_column_names,data_file_name,num_features,training_samples,n_layers,learning_rate,batch_size,activation_function)
print(selected_arch)


input_shape=(128,128,1)
max_epochs=200
patience_epochs=20
metric_to_evaluate="iou"
error_metric=0




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
                                                metric_to_evaluate)

# %%



