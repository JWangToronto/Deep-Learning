'''
Created by Jacky on Mar. 29, 2018
'''
import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from keras import Sequential
from keras.layers  import Dense,Dropout,LSTM

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import load_model
from keras import regularizers

def testing_file(test_file):
    data = read_processing_one_file(data_path + test_file)
    pred = model.predict(data.values)
    data['Predicted_placements'] = [np.argmax(y, axis=None, out=None) for y in pred]
    df = pd.read_csv(data_path + test_file, header=None)
    df.index = df[1]
    df = pd.concat([df, data[['Predicted_placements']]], join='inner', axis=1)
    df = df.reset_index(drop=True)
    df.to_csv(data_path + test_file[:-4] + '-with-predicted-placements-by-DNN.csv', sep=',')
    return

def read_processing_one_file(file):
    data = pd.read_csv(file, header=None)
    data.columns = data_cols

    data['ACCELEROMETER_X'] = data['X'][data['Type'] == 'ACCELEROMETER']
    data['ACCELEROMETER_Y'] = data['Y'][data['Type'] == 'ACCELEROMETER']
    data['ACCELEROMETER_Z'] = data['Z'][data['Type'] == 'ACCELEROMETER']

    data['GYROSCOPE_X'] = data['X'][data['Type'] == 'GYROSCOPE']
    data['GYROSCOPE_Y'] = data['Y'][data['Type'] == 'GYROSCOPE']
    data['GYROSCOPE_Z'] = data['Z'][data['Type'] == 'GYROSCOPE']

    data['GRAVITY_X'] = data['X'][data['Type'] == 'GRAVITY']
    data['GRAVITY_Y'] = data['Y'][data['Type'] == 'GRAVITY']
    data['GRAVITY_Z'] = data['Z'][data['Type'] == 'GRAVITY']

    del data['Type']
    del data['X']
    del data['Y']
    del data['Z']

    data = data.groupby('Time').sum()
    data[data == 0] = np.nan
    for col in data.columns:
        data[col] = data[col].interpolate()
        data[col].bfill(inplace=True)
    return data

np.random.seed(7)

os.chdir("E:/Dr Wang/for job/interviews/")
data_path = "Testcase5_RD_Engineer/"

data_cols = ['Type', 'Time','X', 'Y', 'Z']
data_files_types=['data-ear','data-hand-landscape','data-hand-portrait','data-hand-swinging','data-pocket']
phone_placements=['ear','hand-landscape','hand-portrait','hand-swinging','pocket']
file_number = 3

train =pd.DataFrame()
for file_type_name in data_files_types:
    for file_i in range(1,file_number):
        data=read_processing_one_file(data_path + file_type_name + '-' + str(file_i) + '.csv')
        data['Target']=data_files_types.index(file_type_name)
        train = pd.concat([train,data])

y = train['Target'].values
del train['Target']
X = train.values
print(X[:5],y[:5])
print(X.shape,y.shape)


#way 2 -deep learning

# scaler = preprocessing.StandardScaler()
# scaler.fit(X)
# scaler.transform(X)
y_DNN = to_categorical(y, num_classes=len(data_files_types))
print(y_DNN[:5])
# l2_lambda = 0.0001

model = Sequential()
# kernel_regularizer=regularizers.l2(l2_lambda),
model.add(
    Dense(32, activation='selu', kernel_initializer='lecun_uniform',
          input_dim=X.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(32, activation='selu',
                kernel_initializer='lecun_uniform'))
model.add(Dropout(0.2))
model.add(
    Dense(32, activation='selu',  kernel_initializer='lecun_uniform'))
model.add(Dropout(0.5))
model.add(
    Dense(32, activation='selu',  kernel_initializer='lecun_uniform'))
model.add(Dropout(0.5))
model.add(
    Dense(16, activation='selu',  kernel_initializer='lecun_uniform'))
model.add(
    Dense(len(data_files_types), activation='softmax',  kernel_initializer='lecun_uniform'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(X, y_DNN, epochs=3, batch_size=64, validation_split=0.1)

train_score = model.evaluate(X, y_DNN)
print(train_score)
print(model.summary())
#test the training file---data-hand-landscape-3.csv
testing_file('data-hand-landscape-3.csv')

#test 1
testing_file('test-dataset-1.csv')

# test 2
testing_file('test-dataset-2.csv')

