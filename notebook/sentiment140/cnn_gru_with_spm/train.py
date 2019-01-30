import pandas as pd
import sentencepiece as spm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, CuDNNGRU
from keras.layers import SpatialDropout1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential

def load(
        trainfile="./train.csv",
        testfile="./test.csv",
        spfile="./m.model"
):
    df_train = pd.read_csv(trainfile, header=None, error_bad_lines=False, encoding="latin1")
    df_test = pd.read_csv(testfile, header=None, error_bad_lines=False, encoding="latin1")
    sp = spm.SentencePieceProcessor()
    sp.Load(spfile)
    return shuffle(df_train[[0,5]]), df_test[[0,5]], sp


def preprocessing(df_train, df_test, sp, maxlen=300):
    X, y = pad_sequences([sp.EncodeAsIds(x) for x in df_train[5]], maxlen=maxlen), [int(x) != 0 for x in df_train[0]]    
    X_test, y_test = pad_sequences([sp.EncodeAsIds(x) for x,y in zip(df_test[5],df_test[0]) if int(y) != 2], maxlen=maxlen), [int(x) != 0 for x in df_test[0] if int(x) != 2]
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model(max_features=8000, max_len=300, dropout_rate=0.2, dim=200, gru_size=100):
    model = Sequential()
    model.add(Embedding(max_features+1, dim, input_length=max_len))
    model.add(SpatialDropout1D(dropout_rate))
    model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(CuDNNGRU(gru_size))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    from sklearn.metrics import classification_report, accuracy_score
    from keras.models import load_model
    epochs = 2
    batch_size=1000
    tr, va, te = preprocessing(*load())
    model = build_model()
    model.fit(*tr, validation_data=va, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save("model.h5")
    model = load_model("model.h5")
    y_pred = model.predict_classes(te[0])
    #print(y_pred.shape)
    #print(y_pred[:10])
    print("ACC:",accuracy_score(te[1], y_pred))
    print(classification_report(te[1], y_pred))


    
