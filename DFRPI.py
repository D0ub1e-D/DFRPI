import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K
np.random.seed(1399)
import pandas as pd

from tensorflow.keras import optimizers

data_train = np.float32(np.load('data/Final_Result.npy'))  


encoder_dim11 = 1100  
encoder_dim12 = 800  
encoder_dim13 = 500  
encoder_dim14 = 200  
encoder_dim15 = 47  
input_data= Input(shape=(1399,))

encoded = Dense(encoder_dim11, activation='relu')(input_data)
encoded = Dense(encoder_dim12, activation='relu')(encoded)
encoded = Dense(encoder_dim13, activation='relu')(encoded)
encoded = Dense(encoder_dim14, activation='relu')(encoded)
encoder_output = Dense(encoder_dim15, activation='relu')(encoded)

decoded = Dense(encoder_dim14, activation='relu')(encoder_output)
decoded = Dense(encoder_dim13, activation='relu')(decoded)
encoded = Dense(encoder_dim12, activation='relu')(decoded)
encoded = Dense(encoder_dim11, activation='relu')(decoded)
decoded_output = Dense(1399, activation='relu')(decoded)

autoencoder = Model(inputs=input_data, outputs=decoded_output)
encoder = Model(inputs=input_data, outputs=encoder_output)
autoencoder.compile(optimizer = optimizers.Adam(lr=0.0001), loss='mae', metrics=['accuracy'])
autoencoder.fit(data_train, data_train,
                epochs=10,
                batch_size=32,
                shuffle=True,
                )
encoded_data = encoder.predict(data_train)
print(encoded_data.shape)
df = pd.DataFrame(encoded_data)




import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

import pandas as pd
import numpy as np

data = pd.read_csv("data/a.csv", error_bad_lines=False)
np.save("data/a.npy", data)
print(data.shape)

np.random.seed(1399)
import pandas as pd
encoded_data=np.float32(np.load('data/a.npy'))  

def my_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true) - 1.9345105161344374e-07, axis=-1)

encoded = Dense(encoder_dim11, activation='relu')(input_data)
encoded = Dense(encoder_dim12, activation='relu')(encoded)
encoded = Dense(encoder_dim13, activation='relu')(encoded)
encoded = Dense(encoder_dim14, activation='relu')(encoded)
encoder_output = Dense(encoder_dim15, activation='relu')(encoded)

decoded = Dense(encoder_dim14, activation='relu')(encoder_output)
decoded = Dense(encoder_dim13, activation='relu')(decoded)
encoded = Dense(encoder_dim12, activation='relu')(decoded)
encoded = Dense(encoder_dim11, activation='relu')(decoded)
decoded_output = Dense(1399, activation='relu')(decoded)

autoencoder = Model(inputs=input_data, outputs=decoded_output)
encoder = Model(inputs=input_data, outputs=encoder_output)

autoencoder.compile(optimizer='adam', loss=my_loss, metrics=['accuracy'])
autoencoder.fit(encoded_data, encoded_data,
                 epochs=10,
                 batch_size=32,
                 shuffle=True,
                 )

encoded_data = encoder.predict(encoded_data)
print(encoded_data.shape)
df = pd.DataFrame(encoded_data)





