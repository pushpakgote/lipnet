from keras.layers import Conv3D, ZeroPadding3D,MaxPooling3D, Dense, Activation, SpatialDropout3D, Flatten, Bidirectional, TimeDistributed,LSTM,GRU,BatchNormalization,Input
from keras.models import Model
from keras import backend as K

class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, output_size=28):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.output_size = output_size
        self.build()

    def build(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)
        

        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        self.zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        self.conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(self.zero1)
        self.batc1 = BatchNormalization(name='batc1')(self.conv1)
        self.actv1 = Activation('relu', name='actv1')(self.batc1)
        self.drop1 = SpatialDropout3D(0.5)(self.actv1)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.drop1)

        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.maxp1)
        self.conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(self.zero2)
        self.batc2 = BatchNormalization(name='batc2')(self.conv2)
        self.actv2 = Activation('relu', name='actv2')(self.batc2)
        self.drop2 = SpatialDropout3D(0.5)(self.actv2)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.drop2)

        self.zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.maxp2)
        self.conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(self.zero3)
        self.batc3 = BatchNormalization(name='batc3')(self.conv3)
        self.actv3 = Activation('relu', name='actv3')(self.batc3)
        self.drop3 = SpatialDropout3D(0.5)(self.actv3)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.drop3)

        self.resh1 = TimeDistributed(Flatten())(self.maxp3)

        #self.gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1',reset_after=False), merge_mode='concat')(self.resh1)
        #self.gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2',reset_after=False), merge_mode='concat')(self.gru_1)

        self.lstm1 = Bidirectional(LSTM(256, kernel_initializer='Orthogonal', return_sequences=True) )(self.resh1)
        self.lstm2 = Bidirectional(LSTM(256, kernel_initializer='Orthogonal', return_sequences=True) )(self.lstm1)

        # transforms RNN output to character activations:
        #self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.gru_2)
        self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.lstm2)

        self.y_pred = Activation('softmax', name='softmax')(self.dense1)

        self.model = Model(inputs=self.input_data, outputs=self.y_pred)

