from keras.models import Model,Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.applications.vgg16 import preprocess_input,VGG16
from keras.regularizers import l2


class VGG_Classifier:
    @staticmethod
    def build(width, height, depth, classes):
        vgg = VGG16(input_shape=(height, width, depth), weights='imagenet', include_top=False)

        #Prevent training already trained layers
        for layer in vgg.layers:
            layer.trainable = False
        
        #Add flatten layer
        x = Flatten()(vgg.output)

        #More Dense layers

        #Use weight regularization(L2 vector norm) and dropout layers to reduce overfitting
        x=Dense(1000,activation="relu",kernel_regularizer=l2(0.001))(x)
        x=Dropout(0.5)(x)

        x=Dense(256,activation="relu",kernel_regularizer=l2(0.001))(x)
        x=Dropout(0.5)(x)

        #Dense layer with number of nuerons equals to number of classes.
        prediction = Dense(classes, activation='softmax')(x)

        #Create the model object
        model = Model(inputs=vgg.input, outputs=prediction)
        return model