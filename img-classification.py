from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os,random,cv2


image_aug = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    validation_split=.2
    )

training = image_aug.flow_from_directory(
    './images/',
    target_size=(256,256),
    color_mode='rgb',
    class_mode= 'categorical',
    batch_size=10,
    shuffle=True,
    subset='training'
)

validation = image_aug.flow_from_directory(
    './images/',
    target_size=(256,256),
    color_mode='rgb',
    class_mode= 'categorical',
    batch_size=10,
    shuffle=True,
    subset='validation'
)

custom_nn = keras.Sequential(
    [
        keras.layers.Input(shape=(256,256,3)),
        keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(3,activation='softmax')
    ]
)

custom_nn.compile(optimizer='sgd',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

custom_nn.fit(training,
               validation_data=validation,
               epochs=4)

custom_nn.save('weights.h5')

if __name__ == '__main__':

    weigthed_model = keras.models.load_model('./weights.h5')
    train_loss,train_accuracy = weigthed_model.evalute(training)
    val_loss,val_accuracy = weigthed_model.evalute(validation)

    import os,random,cv2
    # save all subfile image paths in a variable 
    allImgPaths = [os.path.join('./images',subfile,image)
            for subfile in os.listdir('./images')
            for image in os.listdir(f'./images/{subfile}')]
    # randomly choose the path from allImgPaths
    def randImgPath():return random.choice(allImgPaths)
    # call saved weights from our trained model
    weigthed_model = keras.models.load_model('./weights.h5')
    # show list of classes under images file 
    classes = os.listdir('./images/')
    #create subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    for i, ax in enumerate(ax):
        img_path = randImgPath()
        # produce a random rgb image for subplots
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # make a copy of image and resize  the image as our trained model input shape
        resizeimg =np.expand_dims(cv2.resize(np.copy(image),(256,256)),axis=0)
        prediction = weigthed_model.predict(resizeimg)
        label = classes[np.argmax(prediction)]
        img = cv2.imread(img_path)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'Prediction : {label}')
        
    plt.show()
















