# %%
from platform import architecture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import os
import pathlib
# %% [markdown]
# ### Load the datasets

# %%


def create_metadata(data_file_name,dataset_column_names,x_column_names,to_categorical_column_names,metric_name,error_metric=True):
    dataset=pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/"+data_file_name,names=dataset_column_names)
    y=np.zeros(len(dataset))
    
    #CALCULATE THE Y
    if error_metric:
        min_error=dataset.loc[dataset[metric_name].idxmin()][metric_name]
        error_ratio=min_error/dataset[metric_name]
        y=error_ratio
    else:
        max_acc=dataset.loc[dataset[metric_name].idxmax()][metric_name]
        acc_ratio=max_acc/dataset[metric_name]
        y=acc_ratio
    
    x=dataset[x_column_names]
    # if (len(to_categorical_column_names)>1):
    #     raise Exception("to_categorical_column_names >1 not supported")
    # to_cat_column_values=np.asarray(x[to_categorical_column_names[:]]).ravel()
    for to_categorical_column in to_categorical_column_names:
        to_cat_column_values=np.asarray(x[to_categorical_column]).ravel()
        dummies = pd.get_dummies(to_cat_column_values,prefix='',prefix_sep='')
        x=x.drop(to_categorical_column,axis=1)
        x=pd.concat([x,dummies],axis=1)

    # dummies = pd.get_dummies(to_cat_column_values,prefix='',prefix_sep='')
    # x=x.drop(to_categorical_column_names,axis=1)
    # x=pd.concat([x,dummies],axis=1)
    x.head()
    return x,y



# %% [markdown]
# ### Meta learner Functions

# %%
def create_metamodel(x,y):
    regr = RandomForestRegressor(random_state=0)
    regr.fit(x,y)
    return regr

def create_hp_space(num_features,training_samples,n_layers,learning_rate,batch_size,activation_function):
    #CREATE THE HYPERPARAMETER SPACE
    dict_all_hyperparams=dict(num_features=num_features,
                                training_samples=[training_samples],
                                n_layers=n_layers,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                activation_function=activation_function)

    grid_search_population=pd.DataFrame(ParameterGrid(dict_all_hyperparams))
    return grid_search_population
    
def predict_hp_space(grid_search_population,regr,to_categorical_column_names,to_categorical_values,x_column_names):
    #PREPROCESS THE DATA TO BE PREDICTED BY THE METALEARNER
    # if (len(to_categorical_column_names)>1):
    #     raise Exception("to_categorical_column_values >1 not supported")
    to_categorical_values=[]
    for to_categorical_column in to_categorical_column_names:
        to_categorical_values.append(grid_search_population[to_categorical_column].unique())
        to_cat_column_values=np.asarray(grid_search_population[to_categorical_column]).ravel()
        dummies = pd.get_dummies(to_cat_column_values,prefix='',prefix_sep='')
        x_test=pd.concat([grid_search_population[x_column_names],dummies],axis=1)
        x_test=x_test.drop(to_categorical_column,axis=1)
    
    #PREDICTION OF THE HYPERPARAMETER SPACE
    predictions= pd.DataFrame(regr.predict(x_test))
    x_test_predicted=x_test.loc[:]

    #REVERSE THE CATEGORICAL OF THE ACTIVATION FUNCTION
    for to_categorical_column,to_categorical_value in to_categorical_column_names,to_categorical_values:
        x_test_predicted[to_categorical_column]=x_test_predicted[to_categorical_value].idxmax(axis=1)
        x_test_predicted=x_test_predicted.drop(to_categorical_value,axis=1)

    x_test_predicted["y"]=pd.DataFrame(regr.predict(x_test))
    x_test_predicted=x_test_predicted.sort_values("y",ascending=False)
    return x_test_predicted
    
def get_top_hp_combination(n_top_hp_to_select,x_test_predicted):

    #SEARCH FOR THE TOP COMBINATION
    top_lr=[]
    top_bz=[]
    top_layers=[]
    top_af=[]
    search=True
    topi=1
    finish_order=[]
    while(search): 
        if len(top_lr)<n_top_hp_to_select:
            top_lr=x_test_predicted.head(topi)["learning_rate"].unique()
        else:
            if("learning_rate" not in finish_order): finish_order.append("learning_rate")
        
        if len(top_bz)<n_top_hp_to_select:
            top_bz=x_test_predicted.head(topi)["batch_size"].unique()
        else:
            if ("batch_size" not in finish_order): finish_order.append("batch_size")
        
        if len(top_layers)<n_top_hp_to_select:
            top_layers=x_test_predicted.head(topi)["n_layers"].unique()
        else:
            if ("n_layers" not in finish_order): finish_order.append("n_layers")
        
        if len(top_af)<n_top_hp_to_select:
            top_af=x_test_predicted.head(topi)["activation_function"].unique()
        else:
            if ("activation_function" not in finish_order): finish_order.append("activation_function")
        
        topi +=1
        if len(top_lr)<n_top_hp_to_select or len(top_bz)<n_top_hp_to_select or len(top_layers)<n_top_hp_to_select or len(top_af)<n_top_hp_to_select:
            search=True
        else:
            if("learning_rate" not in finish_order): finish_order.append("learning_rate")
            if ("batch_size" not in finish_order): finish_order.append("batch_size")
            if ("n_layers" not in finish_order): finish_order.append("n_layers")
            if ("activation_function" not in finish_order): finish_order.append("activation_function")
            search=False
    
    return top_lr,top_bz,top_layers,top_af,finish_order





# %%

#DATASET NAMES    
def meta_learner(n_top_hp_to_select,dataset_column_names,x_column_names,metric_name,to_categorical_column_names,data_file_name,
                num_features,training_samples,n_layers,learning_rate,batch_size,activation_function):
    

    
    x,y=create_metadata(data_file_name,dataset_column_names,x_column_names,to_categorical_column_names,metric_name,error_metric=True)                              
    
    model=create_metamodel(x,y)
    gs_population=create_hp_space(num_features,training_samples,n_layers,learning_rate,batch_size,activation_function)
    
    predictions=predict_hp_space(gs_population,model,to_categorical_column_names,activation_function,x_column_names)

    top_lr,top_bz,top_layers,top_af,finish_order=get_top_hp_combination(n_top_hp_to_select,predictions)
    return top_lr,top_bz,top_layers,top_af,finish_order



# %%


def test_metalearner(data_file_name,n_top_hp_to_select,dataset_column_names,x_column_names,metric_name,training_samples):
    to_categorical_column_names=["activation_function"]
    
    max_epochs=10
    patience_epochs=2
    metric_to_evaluate="balanced_accuracy"
    sort_order_desc=True
    architecture_name="irnet"
    problem_type="prediction"
    #FILES NAME
    hp_dataset_name="test_hp_dataset.csv"
    weights_folder="data/weights/"
    #HYPERPARAMETERS TO EVALUATE
    num_features=[29]
    training_and_validation_samples=99999
    n_layers=[1,2,3]
    learning_rate=[0.01,0.001,0.0001,0.00001]
    batch_size=[16,32,64,128]
    activation_function=['relu','elu','tanh','sigmoid']

    #GA configuration
    all_hyperparams=[n_layers,learning_rate,batch_size,activation_function]
    population_size=6
    sel_prt=2
    rand_prt=2
    generations=2
    
    meta_learner(n_top_hp_to_select,dataset_column_names,x_column_names,metric_name,to_categorical_column_names,data_file_name,
                num_features,training_samples,n_layers,learning_rate,batch_size,activation_function)
    return 1
    
# %%

def load_metadata(data_file_name,dnn_architecture,dnn_task,dnn_dim):
    dataset=pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/"+data_file_name,names=dataset_column_names)
    selected_arch=""
    
    error_metric= 1 if dnn_dim==1 else 0

    if(dnn_architecture=='all'):
        dataset_task_dim=dataset.loc[(dataset["task"]==dnn_task) &
                                        (dataset["dimension"]==str(dnn_dim)) &
                                        (dataset["error_metric"]==str(error_metric))]
        dataset_task_dim["metric"] = pd.to_numeric(dataset_task_dim["metric"])
        dataset_task_dim=dataset_task_dim.sort_values("metric",ascending=bool(error_metric))
        best_arch=(dataset_task_dim.head(1)).loc[:,"architecture"].values[0]
        dataset_task_dim_arch=dataset_task_dim.loc[(dataset_task_dim["architecture"]==best_arch)]
    else:
        dataset_task_dim_arch=dataset.loc[(dataset["architecture"]==dnn_architecture) &
                                    (dataset["task"]==dnn_task) &
                                    (dataset["dimension"]==str(dnn_dim)) &
                                    (dataset["error_metric"]==str(error_metric))]
    
    
    x=dataset_task_dim_arch.loc[:,x_column_names]
    x=x.reset_index(drop=True)
    y=dataset_task_dim_arch.loc[:,'y']
    for to_categorical_column in to_categorical_column_names:
        to_cat_column_values=np.asarray(x[to_categorical_column]).ravel()
        dummies = pd.get_dummies(to_cat_column_values,prefix='',prefix_sep='')
        x=x.drop(to_categorical_column,axis=1)
        x=x.apply(pd.to_numeric)
        x=pd.concat([x,dummies],axis=1)
    return x,y


n_top_hp_to_select=2
dataset_column_names=["architecture","error_metric","task","num_features",
                        "training_samples","n_layers","activation_function",
                        "learning_rate","batch_size","metric","dimension","dataset","y"]

x_column_names=["num_features","training_samples",
                        "n_layers","activation_function",
                        "learning_rate", "batch_size"]
metric_name="metric"
to_categorical_column_names=["activation_function"]

dnn_architecture="all"
dnn_task="prediction"
dnn_dim=1
training_samples=1999
x,y=load_metadata("data/metadataset.csv",dnn_architecture,dnn_task,dnn_dim)
