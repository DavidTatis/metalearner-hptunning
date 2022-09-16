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
    if error_metric==True:
        min_error=dataset.loc[dataset[metric_name].idxmin()][metric_name]
        error_ratio=min_error/dataset[metric_name]
        y=error_ratio
    else:
        max_acc=dataset.loc[dataset[metric_name].idxmax()][metric_name]
        acc_ratio=dataset[metric_name]/max_acc
        y=acc_ratio
    
    x=dataset[header_metadata]
    x.loc[:,"dimension"]=dim
    x.loc[:,"dataset"]=dataset_name
    xy=pd.concat([x,y],axis=1)
    xy.columns=[*header_metadata,"dimension","dataset",'y']
    # if add_header: pd.DataFrame(columns=header_metadata).to_csv(os.path.dirname(os.path.abspath(__file__))+"/"'data/metadataset.csv', mode='a', header=False,index=False)

    xy.to_csv(os.path.dirname(os.path.abspath(__file__))+"/"+'data/metadata_fixed.csv', mode='a', header=add_header,index=False)

def test(data_file_name,add_header=False,dim=1,dataset_name="empty",error_metric=False):
    dataset_column_names=["architecture","error_metric","task","num_features","training_samples",
                    "n_layers", "input_shape","activation_function",
                    "learning_rate", "batch_size", "epochs","loss","metric"]
    header_metadata=["architecture","error_metric","task","num_features","training_samples","n_layers","activation_function","learning_rate", "batch_size","metric"]
    
    metric_name="metric"

    write_meta_data(data_file_name,dataset_column_names,header_metadata,metric_name,error_metric=error_metric,add_header=add_header,dim=dim,dataset_name=dataset_name)
    



dataset_name="flight-price-prediction"
test("data/1d_irnet.csv",dim=1,dataset_name=dataset_name,add_header=True,error_metric=True)
test("data/1d_fcmnr.csv",dim=1,dataset_name=dataset_name,error_metric=True)
test("data/1d_fcunet.csv",dim=1,dataset_name=dataset_name,error_metric=True)


dataset_name="brain-mri-segmentation"
test("data/2d_mnr.csv",dim=2,dataset_name=dataset_name,error_metric=False)
test("data/2d_unet.csv",dim=2,dataset_name=dataset_name,error_metric=False)
test("data/2d_irnet.csv",dim=2,dataset_name=dataset_name,error_metric=False)

dataset_name="indian-pines"
test("data/3d_cnn_fixed.csv",dim=3,dataset_name=dataset_name,error_metric=False)


# test("data/FIXED:cnn_hyperparams_with_metric.csv",dim=3,add_header=True,dataset_name=dataset_name)
# test("data/1d_irnet.csv",dim=1,dataset_name=dataset_name)
# test("data/1d_fcmnr.csv",dim=1,dataset_name=dataset_name)

# dataset_name="brain-mri-segmentation"
# test("data/2d_mnr.csv",dim=2,dataset_name=dataset_name)
# test("data/2d_unet.csv",dim=2,dataset_name=dataset_name)
# test("data/2d_irnet.csv",dim=2,dataset_name=dataset_name)

# dataset_name="indian-pines"
# test("data/3d_cnn.csv",dim=3,dataset_name=dataset_name)