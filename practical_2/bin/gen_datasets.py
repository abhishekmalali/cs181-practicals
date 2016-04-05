import os
import pandas as pd
import numpy as np
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from scipy import sparse
import util

TRAIN_DIR = "../data/train"
call_set = set([])
def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)

#creating a set of features counting the number of tags
def call_feats(tree, good_calls):
    #Inputs
    #tree - tree object for every file
    #good_calls - list of tags for which we create the features
    call_counter = {}
    for el in tree.iter():
        call = el.tag
        if call not in call_counter:
            call_counter[call] = 0
        else:
            call_counter[call] += 1

    call_feat_array = np.zeros(len(good_calls))
    for i in range(len(good_calls)):
        call = good_calls[i]
        call_feat_array[i] = 0
        if call in call_counter:
            call_feat_array[i] = call_counter[call]
    return call_feat_array


###Creating function for loading data
def create_matrix(start_index, end_index, tags, direc="../data/train"):
    X = None
    classes = []
    ids = []
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue
            
        i += 1
        if i < start_index:
            continue
        if i >= end_index:
            break
        id_str, clas = datafile.split('.')[:2]
        ids.append(id_str)
        #adding target class to training data
        try:
            classes.append(util.malware_classes.index(clas))
        except ValueError:
            assert clas == "X"
            classes.append(-1)
            
        #parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
        this_row = call_feats(tree, tags)
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))
            
    return X, np.array(classes), ids    


#List of unique tags
tags = []
for idx in range(numFiles):
    tree = ET.parse(os.path.join(TRAIN_DIR,fileList[idx]))
    for el in tree.iter():
        call = el.tag
        tags.append(call)
    tags = list(np.unique(tags))
unique_tags = np.unique(tags)
#Converting all tags to 'str' from  numpy.string_
unique_tags = [str(tag) for tag in unique_tags]
X_train, t_train, train_ids = create_matrix(0, numFiles,\
                                            unique_tags, TRAIN_DIR)


features_df = pd.DataFrame(X_train,columns=unique_tags)
features_df['class'] = t_train
features_df['id'] = train_ids
#Saving the features dataframe as a new file
features_df.to_csv('../outputs/features_v1.csv')


TEST_DIR = "../data/test"
testFileList = os.listdir(TEST_DIR)
numTestFiles = len(testFileList)
X_test, t_test, test_ids = create_matrix(0, numTestFiles,\
                                            unique_tags, TEST_DIR)

#Ignoring t_train since there is no response variable 
features_test_df = pd.DataFrame(X_test,columns=unique_tags)
features_test_df['class'] = test_ids
features_test_df.to_csv('../outputs/features_test_v1.csv')
