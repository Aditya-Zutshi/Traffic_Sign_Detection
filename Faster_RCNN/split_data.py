import numpy as np
import pandas as pd
import sys
from sklearn.cross_validation import StratifiedShuffleSplit

full_labels = pd.read_csv(sys.argv[1])
full_labels.columns = ["img", "x1", "y1", "x2", "y2", "id"]
for index, row in full_labels.iterrows():
    new_value = row['img'].split('.')
    
def stratifiedshufflesplit(data, test_size=0.3, thres=1):
    y_less = data.groupby("id").filter(lambda x: len(x) <= thres)
    data = pd.concat([data, y_less], ignore_index=True)
    
    sss = StratifiedShuffleSplit(data['id'], 1, test_size=test_size)
    train_index, test_index =list(*sss)
    xtrain, xtest = data.iloc[train_index], data.iloc[test_index]
    #print(xtest['id'].value_counts())
    return xtrain, xtest

xtrain, xt = stratifiedshufflesplit(full_labels, 0.3, 1)
xtest, xval = stratifiedshufflesplit(xt, 0.5, 1)

#len(xtrain), len(xval), len(xtest)

xtrain['width'] = 1360
xtrain['height'] = 800
xtest['width'] = 1360
xtest['height'] = 800
xval['width'] = 1360
xval['height'] = 800
  
xtrain.to_csv((sys.argv[2]+'/train.csv'), index=False)
xtest.to_csv((sys.argv[2]+'/test.csv'), index=False)
xval.to_csv((sys.argv[2]+'/val.csv'), index=False)