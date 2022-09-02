from platform import architecture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import os
import pathlib



def write_meta_data(data_file_name,dataset_column_names,header_metadata,metric_name,error_metric=True,add_header=False,dim=1,dataset_name="empty"):
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
    
    x=dataset[header_metadata]
    x.loc[:,"dimension"]=dim
    x.loc[:,"dataset"]=dataset_name
    xy=pd.concat([x,y],axis=1)
    xy.columns=[*header_metadata,"dimension","dataset",'y']
    # if add_header: pd.DataFrame(columns=header_metadata).to_csv(os.path.dirname(os.path.abspath(__file__))+"/"'data/metadataset.csv', mode='a', header=False,index=False)

    xy.to_csv(os.path.dirname(os.path.abspath(__file__))+"/"'data/metadataset.csv', mode='a', header=add_header,index=False)

def test(data_file_name,add_header=False,dim=1,dataset_name="empty"):
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

    #META DATASET
    n_top_hp_to_select=2
    dataset_column_names=["architecture","error_metric","task","num_features","training_samples",
                    "n_layers", "input_shape","activation_function",
                    "learning_rate", "batch_size", "loss","fit_time","metric"]
    header_metadata=["architecture","error_metric","task","num_features","training_samples","n_layers","activation_function","learning_rate", "batch_size","metric"]
    x_column_names=["architecture","num_features","training_samples",
                            "n_layers","activation_function",
                            "learning_rate", "batch_size"]
    metric_name="metric"
    to_categorical_column_names=["activation_function","architecture"]
    

    write_meta_data(data_file_name,dataset_column_names,header_metadata,metric_name,error_metric=True,add_header=add_header,dim=dim,dataset_name=dataset_name)
    # top_lr,top_bz,top_layers,top_af,finish_order=meta_learner(2,dataset_column_names,x_column_names,metric_name,to_categorical_column_names,data_file_name,
    #                                                             num_features,training_and_validation_samples,n_layers,learning_rate,batch_size,activation_function)
    # print(top_lr,top_bz,top_layers,top_af,finish_order)
    



dataset_name="flight-price-prediction"
test("data/1d_fcunet.csv",dim=1,add_header=True,dataset_name=dataset_name)
test("data/1d_irnet.csv",dim=1,dataset_name=dataset_name)
test("data/1d_fcmnr.csv",dim=1,dataset_name=dataset_name)

dataset_name="brain-mri-segmentation"
test("data/2d_mnr.csv",dim=2,dataset_name=dataset_name)
test("data/2d_unet.csv",dim=2,dataset_name=dataset_name)
test("data/2d_irnet.csv",dim=2,dataset_name=dataset_name)

dataset_name="indian-pines"
test("data/3d_cnn.csv",dim=3,dataset_name=dataset_name)