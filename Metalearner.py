# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

# %% [markdown]
# ### Load the datasets

# %%
def load_meta_data(data_file_name,dataset_column_names,x_column_names,to_categorical_column_names,metric_name,error_metric=True):
    dataset=pd.read_csv(data_file_name,names=dataset_column_names)
    y=np.zeros(len(dataset))
    print(len(y))
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
    print(x[to_categorical_column_names[:]])
    dummies = pd.get_dummies(x[to_categorical_column_names[:]],prefix=[''],prefix_sep=[''])
    x=x.drop(to_categorical_column_names,axis=1)
    x=pd.concat([x,dummies],axis=1)
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
                                training_samples=training_samples,
                                n_layers=n_layers,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                activation_function=activation_function)

    grid_search_population=pd.DataFrame(ParameterGrid(dict_all_hyperparams))
    return grid_search_population
    
def predict_hp_space(grid_search_population,regr,to_categorical_column_names,to_categorical_values,x_column_names):
    #PREPROCESS THE DATA TO BE PREDICTED BY THE METALEARNER
    dummies2 = pd.get_dummies(grid_search_population[to_categorical_column_names[:]],prefix=[''],prefix_sep=[''])
    x_test=pd.concat([grid_search_population[x_column_names],dummies2],axis=1)
    x_test=x_test.drop(to_categorical_column_names,axis=1)
    #PREDICTION OF THE HYPERPARAMETER SPACE
    predictions= pd.DataFrame(regr.predict(x_test))
    x_test_predicted=x_test.loc[:]

    #REVERSE THE CATEGORICAL OF THE ACTIVATION FUNCTION
    x_test_predicted["activation_function"]=x_test_predicted[to_categorical_values].idxmax(axis=1)
    x_test_predicted=x_test_predicted.drop(to_categorical_values,axis=1)

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
def meta_learner(n_top_hp_to_select):
    dataset_column_names=["architecture","task","num_features","training_samples",
                "n_layers", "input_shape","activation_function",
                "learning_rate", "batch_size", "loss","fit_time","mae"]
    x_column_names=["num_features","training_samples",
                            "n_layers","activation_function",
                            "learning_rate", "batch_size",]
    metric_name="mae"
    to_categorical_column_names=["activation_function"]
    data_file_name="./data/1d_irnet.csv"
    #HYPERPARAMETERS TO EVALUATE
    num_features=[29]
    training_samples=[240122]
    n_layers=[1,2,3]
    learning_rate=[0.01,0.001,0.0001,0.00001]
    batch_size=[16,32,64,128]
    activation_function=['relu','elu','tanh','sigmoid']



    x,y=load_meta_data(data_file_name,dataset_column_names,x_column_names,to_categorical_column_names,metric_name,error_metric=True)                              
    model=create_metamodel(x,y)
    gs_population=create_hp_space(num_features,training_samples,n_layers,learning_rate,batch_size,activation_function)
    predictions=predict_hp_space(gs_population,model,to_categorical_column_names,activation_function,x_column_names)
    top_lr,top_bz,top_layers,top_af,finish_order=get_top_hp_combination(n_top_hp_to_select,predictions)
    return top_lr,top_bz,top_layers,top_af,finish_order


# %%



