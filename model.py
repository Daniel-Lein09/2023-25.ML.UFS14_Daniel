import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.src.layers import LSTM, Dense
from keras.src.models import Sequential
from keras.src.optimizers import Adam

#Scarico i dati dell'indice tramite la funzione download di yahoo finance, indicando anche l'inizio e la fin

sp_data = yf.download("^GSPC", start="2000-01-01", end="2014-01-01").reset_index()
sp_data

selected_col = ["Date", "Adj Close"]

selected_sp = sp_data[selected_col]
selected_sp

plt.plot(selected_sp["Date"], selected_sp["Adj Close"])

selected_period_train = selected_sp[(sp_data["Date"] >= "2000-01-01") & (selected_sp["Date"] <= "2005-12-31")]
selected_period_train

x_pre_train = selected_period_train["Adj Close"].to_numpy().reshape(-1,1)
x_pre_train.shape


selected_period_test = selected_sp[(selected_sp["Date"] >= "2006-01-01") & (selected_sp["Date"] <= "2006-12-31")]
selected_period_test

x_pre_test = selected_period_test["Adj Close"].to_numpy().reshape(-1,1)
x_pre_test.shape

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(x_pre_train)
scaled_train.shape

scaled_test = scaler.fit_transform(x_pre_test)
scaled_test.shape

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


time_step = 60

x_train, y_train = create_sequences(scaled_train, 60)
x_train

x_test, y_test = create_sequences(scaled_test, 60)
y_test.shape

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


model = Sequential(
    layers=(
        LSTM(100, activation= "sigmoid", return_sequences=True, input_shape=(60, 1), ),
        LSTM(50, activation="relu", return_sequences=False), 
        Dense(1)
    )
)

model.compile(optimizer="Adam", loss='mean_absolute_error')
history = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_test, y_test))

# Previsione e trasformazione inversa dei dati
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


def plot_graphs():
    # Grafico della serie temporale
    plt.figure(figsize=(14, 5))
    plt.plot(selected_period_train["Date"], x_pre_train, label="Training Data")
    plt.plot(selected_period_test["Date"], x_pre_test, label="Test Data")
    plt.title("Serie temporale dei dati di training e test")
    plt.xlabel("Data")
    plt.ylabel("Variazione Percentuale")
    plt.legend()
    plt.show()

    # Grafico delle perdite di addestramento e validazione
    plt.figure(figsize=(14, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Perdite di addestramento e validazione')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Predizioni del modello sui dati di test
    predictions = model.predict(x_test)
    plt.figure(figsize=(14, 5))
    plt.plot(range(len(y_test)), y_test, label='Valori Reali')
    plt.plot(range(len(predictions)), predictions, label='Predizioni')
    plt.title('Predizioni del modello sui dati di test')
    plt.xlabel('Giorni')
    plt.ylabel('Variazione Percentuale Normalizzata')
    plt.legend()
    plt.show()

plot_graphs()