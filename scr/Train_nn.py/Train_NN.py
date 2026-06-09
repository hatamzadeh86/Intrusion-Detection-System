from dataclasses import dataclass  , asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , recall_score,precision_score ,root_mean_squared_error, confusion_matrix , mean_squared_error , r2_score ,accuracy_score,log_loss,f1_score
from typing import Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ,cross_val_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder , StandardScaler ,LabelEncoder
import optuna
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset , DataLoader
import torch.optim as optim




X = uf_clean[['dst_host_srv_count', 'dst_host_same_srv_rate', 'flag',
       'logged_in', 'same_srv_rate', 'dst_host_rerror_rate', 'rerror_rate',
       'srv_rerror_rate', 'dst_host_srv_rerror_rate', 'service',
       'dst_host_count', 'dst_host_srv_serror_rate', 'dst_host_serror_rate',
       'serror_rate', 'srv_serror_rate', 'count', 'diff_srv_rate',
       'dst_host_diff_srv_rate', 'protocol_type', 'is_guest_login',
       'num_failed_logins', 'duration', 'srv_count', 'dst_bytes',
       'srv_diff_host_rate', 'hot', 'dst_host_same_src_port_rate',
       'wrong_fragment', 'num_access_files', 'num_shells', 'is_host_login',
       'urgent', 'land', 'dst_host_srv_diff_host_rate', 'src_bytes',
       'root_shell', 'su_attempted']].to_numpy()

Y = uf_clean['target_ATTACK'].to_numpy()
#_______________________________________________________________________________________________________________
X_work , test_X , Y_work , test_Y = train_test_split(X , Y , test_size=0.2 , random_state=42 , stratify=Y)

train_X , val_X , train_Y , val_Y = train_test_split(X_work , Y_work , test_size=0.2 , random_state=42 ,stratify=Y_work)
#_______________________________________________________________________________________________________________
# to tensor  :

train_X = torch.tensor(train_X , dtype=torch.float32)
test_X = torch.tensor(test_X , dtype=torch.float32)
val_X = torch.tensor(val_X , dtype=torch.float32)

train_Y = torch.tensor(train_Y , dtype=torch.long)
test_Y = torch.tensor(test_Y , dtype=torch.long)
val_Y = torch.tensor(val_Y , dtype=torch.long)
#_______________________________________________________________________________________________________________
from torch.utils.data import TensorDataset , dataloader

train_dataset = TensorDataset(train_X , train_Y)
test_dataset = TensorDataset(test_X , test_Y)
val_dataset = TensorDataset(val_X , val_Y)

train_loader = DataLoader(train_dataset , batch_size=32 , shuffle=True)
test_loader = DataLoader(test_dataset , batch_size=32)
val_loader = DataLoader(val_dataset , batch_size=32)

#_______________________________________________________________________________________________________________
# modeling :

class NET_model (nn.Module):
    def __init__(self , hidden_size):
        super().__init__()

        self.layer1 = nn.Linear(37 , hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size , hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size , hidden_size)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden_size , hidden_size)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Linear(hidden_size , hidden_size)
        self.relu5 = nn.ReLU()
        self.layer6 = nn.Linear(hidden_size , 1)

    def forward (self , x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.relu4(x)
        x = self.layer5(x)
        x = self.relu5(x)
        x = self.layer6(x)
        return x
#_______________________________________________________________________________________________________________

def objectiv (trial):

    hidden_size = trial.suggest_int('hidden_size' , 30 , 250)
    Lr = trial.suggest_float('Lr' , 1e-5 , 1e-1 , log=True)

    model = NET_model(hidden_size)
    optimizer = optim.Adam(model.parameters() , lr=Lr)
    criterion = nn.BCEWithLogitsLoss()

#___________________________________Train loop_________________________________________________________________
    epochs = 20
    for R in range(epochs):

        model.train()

        for batch_X , batch_Y in train_loader :

            optimizer.zero_grad()
            predict = model(batch_X)
            loss = criterion(predict , batch_Y.float().view(-1 ,1))#-> متریک لاس ام میگه حتما باید تارگت ام از نوع فلوت باشه

            loss.backward()
            optimizer.step()

#_________________________________________eval loop with val_data_____________________________________________________________________

        model.eval()

        all_prds = []
        all_tures = []

        with torch.no_grad():

            for v_batch_X , v_batch_Y in val_loader :

                output = model(v_batch_X) #---->  خروجی مدلم عدد های logics خام
                proba = torch.sigmoid(output) #----> تبدیل اعداد خام به احتمالات
                prdss = (proba > 0.5).float() #---> تبدیل احتمالات به prds float=> 0.0 and 1.0


                all_prds.extend(prdss.cpu().numpy()) # .flatten()
                all_tures.extend(v_batch_Y.cpu().numpy()) # .flatten()

        
        # acc = accuracy_score(tures , prds)
        f1 = f1_score(all_tures , all_prds)

        return  f1

#_______________________________________________________________________________________________________________@
    

 
sampler = optuna.samplers.TPESampler(seed=42  , n_startup_trials=5)
study = optuna.create_study(direction='maximize' , sampler=sampler)

study.optimize(objectiv , n_trials=5 , n_jobs=-1)


best_params = study.best_params

fin_model = NET_model(best_params['hidden_size'])

optimiz = optim.Adam(fin_model.parameters() , lr=best_params['Lr'])
#____________________________________________Train loop finall and Early Stopping + train loss_________________________________________________________________

train_loss = [] # ....
val_loss = []  # برای رسم نمودار خطا هام

# training loop finall :

epochs = 60

patience = 10

best_val_loss = float('inf')
counter = 0
criterion = nn.BCEWithLogitsLoss()

for i in range(epochs):
   runing_loss = 0.0
   
   fin_model.train()

   for batch_X , batch_Y in train_loader :

    optimiz.zero_grad()
    
    output = fin_model(batch_X)
    

    loss = criterion(output , batch_Y.float().view(-1 , 1))

    loss.backward()

    optimiz.step()

    runing_loss += loss.item()


   epoch_loss = runing_loss / len(train_loader)
   train_loss.append(epoch_loss)

#____________________________________________ evaliation finall + val loss __________________________________________________________________

# evaliation :

   fin_model.eval()

   v_prds , v_trues = [] , []

   runing_val_loss = 0.0

   with torch.no_grad():


       for v_batch_X , v_batch_Y in val_loader :

            output_model_vals = fin_model(v_batch_X)
            v_loss = criterion(output_model_vals , v_batch_Y.float().view(-1 , 1)) # برای ارزیابی لاس ام باید فلوت باشد

            runing_val_loss += v_loss.item()

            v_proba = torch.sigmoid(output_model_vals)

            val_Prd = (v_proba > 0.5).float()

            v_prds.extend(val_Prd.cpu().numpy())
            v_trues.extend(v_batch_Y.cpu().numpy())
        
#_______________________________________________ Early Stopping ________________________________________________________________

   current_epoch_val_loss = runing_val_loss / len(val_loader)
   val_loss.append(current_epoch_val_loss)




   if current_epoch_val_loss < best_val_loss :

    best_val_loss = current_epoch_val_loss

    torch.save(fin_model.state_dict() , 'best_model.pth')
    counter = 0


   else :
    counter += 1
    if counter >= patience :
        print('spot train')
        break


#_________________________________________________ finall test data ______________________________________________________________
          

fin_model.eval()
test_prds = []
test_ture = []

with torch.no_grad():

    for t_batch_X , t_batch_Y in test_loader :

        output_model_finall = fin_model(t_batch_X)
        proba = torch.sigmoid(output_model_finall)
        prds_finall = (proba > 0.5).float() # --> 0.0 or 1.0

        test_prds.extend(prds_finall.cpu().numpy().flatten()) 
        test_ture.extend(t_batch_Y.cpu().numpy().flatten())



print('/n' + '_'*50)
print(classification_report(test_ture , test_prds ,target_names=['normal' , 'target_ATTACK']))
print('confsuion_matrix :' , confusion_matrix(test_ture , test_prds))

#_________________________________________________ رسم نمودار خطا ها ______________________________________________________________

plt.figure(figsize=(12,8))

plt.plot(train_loss , label='Train Loss' , color='blue')
plt.plot(val_loss , label='val Loss' , color='red')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title('Train Loss vs val Loss')
plt.legend()
plt.show()







