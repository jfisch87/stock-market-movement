

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import to_categorical
import pickle
import os

import matplotlib.pyplot as plt



def build_model(df, ticker):
    
    # nas/split
    df.dropna(inplace=True)
    X = df.drop(columns=['target', 'ticker', 'price open', 'price close', 'price low'])
    y = df['target']
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=.2)
    
    #scale
    ss = StandardScaler()
    X_train_sc = ss.fit_transform(X_train)
    X_test_sc = ss.transform(X_test)
    
    # time series gen
    tsg_len = 5
    tsg_batch = 512
    train_seq = TimeseriesGenerator(X_train_sc, y_train, 
                                    length=tsg_len, batch_size=tsg_batch)
    test_seq = TimeseriesGenerator(X_test_sc, y_test, 
                                   length=tsg_len, batch_size=tsg_batch)
    # Design RNN
    model = Sequential()
    model.add(GRU(32,
                 input_dim=X.shape[1],
                 return_sequences=True)) # True if next layer is RNN
    model.add(GRU(16,return_sequences=False)) # False if next layer is Dense
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    # output layer
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics = ['accuracy'])
    hist = model.fit(train_seq,
                    epochs=100,
                    validation_data=test_seq,
                    verbose = 0)
    

    plot_acc(hist, ticker)
    plot_loss(hist, ticker)

    # metrics:
    # https://stackoverflow.com/questions/54875846/how-to-print-labels-and-column-names-for-confusion-matrix
    preds = np.argmax(model.predict(test_seq), axis=-1)
    labels = ['Down', 'Flat', 'Up']
    y_cats = np.argmax(y_test, axis=1)
    cf = confusion_matrix(y_cats[tsg_len:], preds)
    cf_df = pd.DataFrame(cf, columns=labels, index=labels)
    cf_df.to_csv(f'./charts/rnn/{resample}/cm/{ticker}.csv', index=True)
    #pickle model
    model.save(f'./models/rnn/{resample}/{ticker}_rnn')
    return hist
#  to load model back:
# from tensorflow import keras
# model = keras.models.load_model('path/to/location')



def plot_acc(hist, ticker):
    # plot acc and loss
    plt.figure()
    plt.plot(hist.history['accuracy'], label = 'Train Accuracy')
    plt.plot(hist.history['val_accuracy'], label = 'Test Accuracy')
    plt.title(label = f'{ticker.upper()} Accuracy', fontsize=16)
    plt.legend()
#     plt.show()
    plt.savefig(f'./charts/rnn/{resample}/acc/{ticker}.png', bbox_inches='tight')
    plt.savefig(f'./charts/rnn/{resample}/acc/{ticker}t.png', transparent=True)
    plt.close()

def plot_loss(hist, ticker):
    plt.figure()
    plt.plot(hist.history['loss'], label = 'Train Loss')
    plt.plot(hist.history['val_loss'], label = 'Test Loss')
    plt.title(label = f'{ticker.upper()} Loss', fontsize=16)
    plt.legend()
#     plt.show()
    plt.savefig(f'./charts/rnn/{resample}/loss/{ticker}.png', bbox_inches='tight')
    plt.savefig(f'./charts/rnn/{resample}/loss/{ticker}t.png', transparent=True)
    plt.close()

results = pd.DataFrame(columns=['loss', 'accuracy', 'val_loss', 'val_accuracy', 'company', 'epoch'])
directory = './datasets/dow_clean/sec/'
files = [file for file in os.listdir(directory) if file.endswith('.csv')]
resample = 'sec'
stocks = {}
counter = 0
for file in files:
    df = pd.read_csv(directory + file, index_col='date_time', parse_dates=True)
    df.sort_index(inplace=True)
    df['target'] = np.where(df['mean_px_1'] > 0, 2,
                           np.where(df['mean_px_1'] < 0, 0, 1))
    ticker = df['ticker'][0]
    # stocks[ticker] = df


# for ticker, df in stocks.items():
    hist = build_model(df, ticker)
    histdf = pd.DataFrame(hist.history)
    histdf['company'] = ticker
    histdf['epoch'] = histdf.index+1
    results.append(histdf)
    counter += 1
    print(f'finished {ticker}. {counter}/{len(files)}')
results.to_csv(f'./charts/rnn/{resample}_results.csv', index=False)