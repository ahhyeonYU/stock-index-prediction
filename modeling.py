import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
import os
from feature_selection import get_data
import FinanceDataReader as fdr
import datetime

#%matplotlib inline

warnings.filterwarnings('ignore')

#plt.rcParams['font.family'] = 'NanumGothic'

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def run_model(base_dir):
    weight_name = os.path.join('tmp_checkpoint.h5')
    df, scaler_kospi = get_data(base_dir)
    #df
    
    window_size = 10

    # bilstm 모델이 필요로하는 input 형식으로 데이터셋 구성
    def make_dataset(data, label, window_size):
        feature_list = []
        label_list = []
        for i in range(len(data) - window_size):
            feature_list.append(np.array(data.iloc[i:i+window_size]))
            label_list.append(np.array(label.iloc[i+window_size-1])) # -1을 한 이유: 이미 한달 뒤 변동 여부가 KOSPI_BINARY 칼럼에 값으로 들어가 있음
        return np.array(feature_list), np.array(label_list)
    
    # 종속변수 정의
    target_col = 'KOSPI_BINARY'
    
    # 독립변수, 종속변수별 데이터셋 분리
    feature_cols = list(df.columns)
    feature_cols.remove(target_col)
    label_cols = [target_col]

    # train, validation 데이터셋 구축
    train_feature = df[feature_cols]
    train_label = df[label_cols]
    train_feature, train_label = make_dataset(train_feature, train_label, window_size)
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
    #x_train.shape, x_valid.shape

    # 모델링
    model = Sequential()
    model.add(Bidirectional(LSTM(16, 
                input_shape=(train_feature.shape[1], train_feature.shape[2]), 
                activation='sigmoid',
                return_sequences=False))
            )

    model.add(Dense(1))
    loss = Huber()
    optimizer = Adam(0.0005)
    model.compile(loss=loss, optimizer=optimizer)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(weight_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    history = model.fit(x_train, y_train, 
                        epochs=200, 
                        batch_size=16,
                        validation_data=(x_valid, y_valid), 
                        callbacks=[early_stop, checkpoint])