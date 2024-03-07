import time
import psutil
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

df = pd.read_csv('MOSFET Modelling Data set2.csv')

conditions = [df['vgs'] <= df['VTO'],df['vgs'] > df['VTO']]
choices = ['Cutoff', 'Other',]
df['Region'] = np.select(conditions, choices)
grouped = df.groupby(df.Region)
df_cutoff = grouped.get_group("Cutoff")
df_Other = grouped.get_group("Other")
df_cutoff["y_calc"]= pow(10,-9)*(df_cutoff['W']/df_cutoff['L'])*pow(10,((df_cutoff['vgs']-df_cutoff['VTO'])/0.06))

conditions1 = [(df_Other['vds'] <= (df_Other['vgs']- df_Other['VTO'])),
              (df_Other['vds'] > (df_Other['vgs']- df_Other['VTO']))]
choices1 = ['Ohmic', 'Saturation']
df_Other['Region'] = np.select(conditions1, choices1)
grouped = df_Other.groupby(df_Other.Region)
df_ohmic = grouped.get_group("Ohmic")
df_saturation = grouped.get_group("Saturation")
df_saturation["y_calc"]= pow(10,-5)*(df_saturation['W']/df_saturation['L'])*(pow((df_saturation['vgs']-df_saturation['VTO']),2))
df_ohmic["y_calc"]= pow(10,-5)*(df_ohmic['W']/df_ohmic['L'])*((2*df_ohmic['vds']*(df_ohmic['vgs']-df_ohmic['VTO']))-pow(df_ohmic['vds'],2))

Cutoff_X = pd.get_dummies(df_cutoff.drop(['Ids','Region','y_calc'], axis=1))
Cutoff_y = df_cutoff.loc[:,"Ids"]
Ohmic_X = pd.get_dummies(df_ohmic.drop(['Ids','Region','y_calc'], axis=1))
Ohmic_y = df_ohmic.loc[:,"Ids"]
Sat_X = pd.get_dummies(df_saturation.drop(['Ids','Region','y_calc'], axis=1))
Sat_y = df_saturation.loc[:,"Ids"]

Cut_X_train, Cut_X_test, Cut_y_train, Cut_y_test = train_test_split(Cutoff_X, Cutoff_y, test_size=.2)
Cut_X_train, Cut_X_val, Cut_y_train, Cut_y_val = train_test_split(Cut_X_train,Cut_y_train, test_size=.1)
ohm_X_train, ohm_X_test, ohm_y_train, ohm_y_test = train_test_split(Ohmic_X, Ohmic_y, test_size=.2)
ohm_X_train, ohm_X_val, ohm_y_train, ohm_y_val = train_test_split(ohm_X_train,ohm_y_train, test_size=.1)
sat_X_train, sat_X_test, sat_y_train, sat_y_test = train_test_split(Sat_X, Sat_y, test_size=.2)
sat_X_train, sat_X_val, sat_y_train, sat_y_val = train_test_split(sat_X_train,sat_y_train, test_size=.1)

print("1")

def Cutoff_region_model(input_dim,layers=np.array([32,64,1])):
    model = Sequential()  
    model.add(Dense(units=layers[0], activation='relu', input_dim=input_dim))
    model.add(Dense(units=layers[1], activation='relu'))
    model.add(Dense(units=layers[2], activation=None))
    return model


def train_the_model(Model,x_train,y_train,x_val,y_val,region,hyperpram,x_test,y_test,OA):    
    #Model.summary()
    if OA=="Adagrad":
        opt=keras.optimizers.Adagrad(learning_rate=hyperpram[2])
    elif OA=="SGD":
        opt=keras.optimizers.SGD(learning_rate=hyperpram[2])
    elif OA=="Adam":
        opt=keras.optimizers.Adam(learning_rate=hyperpram[2])

        
    
    if region=="Cutoff":
        y= pow(10,-9)*(x_train['W']/x_train['L'])*pow(10,((x_train['vgs']-x_train['VTO'])/0.06))
    elif region=="Ohmic":
        y= pow(10,-5)*(x_train['W']/x_train['L'])*((2*x_train['vds']*(x_train['vgs']-x_train['VTO']))-pow(x_train['vds'],2))
    else:
        y= pow(10,-5)*(x_train['W']/x_train['L'])*(pow((x_train['vgs']-x_train['VTO']),2))
        
    def my_loss_fn(y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1
        
    Model.compile(loss=my_loss_fn, optimizer=opt, metrics='MeanSquaredError')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=50,restore_best_weights=True ) 
                            
    
    history=Model.fit(x_train, y_train, epochs=hyperpram[0], batch_size=hyperpram[1], validation_data=(x_val, y_val),verbose=0,callbacks=[early_stopping])
    
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,(len(loss_train)+1))
    #print(loss_train)
    #print(epochs)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss for ' + region + ' region')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    acc_train = history.history['mean_squared_error']
    acc_val = history.history['val_mean_squared_error']
    epochs = range(1,(len(loss_train)+1))
    plt.plot(epochs, loss_train, 'g', label='Training MSE')
    plt.plot(epochs, loss_val, 'b', label='validation MSE')
    plt.title('Training and Validation MSE for ' + region + ' region')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
    score = Model.evaluate(x_test, y_test, verbose=2)
    pred=Model.predict(x_test)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    Model.save('Trained_DNN_model')
    return score[0]

def optimizer(layers,OA):
    detailsx=[]
    for i, layers in enumerate(layers):
        details=[]
        # Measure memory usage before training
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # in MB

        # Start measuring time
        start_time = time.time()
        details.append(layers)
        model=Cutoff_region_model(6,layers)
        #model.summary()
        score=train_the_model(model,sat_X_train,sat_y_train,sat_X_val,sat_y_val,"Omic",cut_hyperpram,sat_X_test,sat_y_test,OA)
        #display(score)
        details.append(score)
        # Stop measuring time
        end_time = time.time()

        # Measure memory usage after training
        memory_after = process.memory_info().rss / (1024 * 1024)  # in MB

        # Calculate time and memory usage
        training_time = end_time - start_time
        memory_usage = memory_after - memory_before
        details.append(training_time)
        details.append(memory_usage)
        detailsx.append(details)
    x=np.array(detailsx,dtype=object)
    x=pd.DataFrame(x,columns=['Layers', 'Test Loss score', 'Training Time(s)', 'Memory Usage(MB)']) 
    mean_test_loss=x['Test Loss score'].mean()
    mean_training_time=x['Training Time(s)'].mean()
    mean_memory_usage=x['Memory Usage(MB)'].mean()
    x = tabulate(x, headers='keys', tablefmt='pretty')
    print("No of epochs: " + str(cut_hyperpram[0]))
    print("Batch size: " + str(cut_hyperpram[1]))
    print("Learning rate: " + str(cut_hyperpram[2]))
    print("Optimization Algorithm: " + OA)
    print(x)
    print("Average Test Loss score: " + str(mean_test_loss))
    print("Average Training Time: " + str(mean_training_time)+ "s")
    print("Avearge Memory Usage: " + str(mean_memory_usage)+ "MB")
    
hp=[[10,8,0.0001,"Adam"]]
for i in range(1):  
    cut_hyperpram=hp[i]
    layers=[[512,512,1]]
    op=optimizer(layers,cut_hyperpram[3])