import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler ,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , recall_score,precision_score , confusion_matrix ,accuracy_score , log_loss , f1_score





# model RF:

def load_data (self , path):
    df = pd.read_csv(path)
    return df

def buil_xy (df):
    X = df[['dst_host_srv_count', 'dst_host_same_srv_rate', 'flag',
       'logged_in', 'same_srv_rate', 'dst_host_rerror_rate', 'rerror_rate',
       'srv_rerror_rate', 'dst_host_srv_rerror_rate', 'service',
       'dst_host_count', 'dst_host_srv_serror_rate', 'dst_host_serror_rate',
       'serror_rate', 'srv_serror_rate', 'count', 'diff_srv_rate',
       'dst_host_diff_srv_rate', 'protocol_type', 'is_guest_login',
       'num_failed_logins', 'duration', 'srv_count', 'dst_bytes',
       'srv_diff_host_rate', 'hot', 'dst_host_same_src_port_rate',
       'wrong_fragment', 'num_access_files', 'num_shells', 'is_host_login',
       'urgent', 'land', 'dst_host_srv_diff_host_rate', 'src_bytes',
       'root_shell', 'su_attempted']]

    Y = df['target_ATTACK']

    return X , Y

def split_data ( X , Y ):
    train_X , test_X , train_Y , test_Y = train_test_split(X , Y , test_size=0.2 , random_state=42)

    return train_X , test_X , train_Y , test_Y

def standar_data (train_X , test_X):
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    return train_X , test_X , scaler

def train_model ( train_X , train_Y):
    model = RandomForestClassifier(n_estimators=300 , criterion='entropy',max_depth=None,min_samples_leaf=2,
                                   min_samples_split=6 ,max_samples=0.7,random_state=42,max_features='log2')
    model.fit(train_X , train_Y)

    return model 

def evalation ( model , test_X , test_Y , train_X , train_Y):

    prds_test = model.predict(test_X)
    prds_train = model.predict(train_X)

    print('acc train :' , accuracy_score(train_Y , prds_train))
    print('acc test :' , accuracy_score(test_Y , prds_test))
    print('recall :' , recall_score(test_Y , prds_test))
    print('precision :' , precision_score(test_Y , prds_test))
    print('confusion matrix :' , confusion_matrix(test_Y , prds_test))
    print('report all metrics :' , classification_report(test_Y , prds_test))

    return prds_test




X , Y = buil_xy(uf_clean) #--- add data cleaning 

train_X , test_X , train_Y , test_Y = split_data(X , Y)

train_X , test_X , scaler = standar_data(train_X , test_X)

model = train_model(train_X , train_Y)

prds_test  = evalation(model , test_X , test_Y , train_X , train_Y)