import numpy as np
from keras import backend as K
from keras.applications import inception_v3

from src.Utils import Utils


class Keras:
    def __init__(self, classes_csv_path, img):
        self.classname_to_id = Utils.load_classes_csv(classes_csv_path)
        print('Initializing Inception V3 model')
        self.model = inception_v3.InceptionV3()
        self.grab_cost_from_model = self.init_keras_model_on_image(img)
        self.class_name_to_fake = ''

    def init_keras_model_on_image(self, img):
        # Load pre-trained image recognition model

        self.class_name_to_fake = self.get_prediction(img)
        # Grab a reference to the first and last layer of the neural net
        model_input_layer = self.model.layers[0].input
        model_output_layer = self.model.layers[-1].output

        # Define the cost function.
        # Our 'cost' to minimize will be the likelihood out image is the original class according to the pre-trained model
        cost_function = model_output_layer[0, self.classname_to_id[self.class_name_to_fake]]

        # Create a Keras function that we can call to calculate the current cost
        cost_function = K.function([model_input_layer, K.learning_phase()],
                                   [cost_function])

        print('Model initialized')
        return cost_function

    def get_prediction(self, img):
        # Load pre-trained image recognition model
        # Add a 4th dimension for batch size (as Keras expects)
        img_array = np.expand_dims(img, axis=0)
        # Run the image through the neural network
        predictions = self.model.predict(img_array)
        # Convert the predictions into text and print them
        predicted_classes = inception_v3.decode_predictions(predictions, top=1)
        imagenet_id, name, confidence = predicted_classes[0][0]
        print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))
        return name

    def get_prediction_on_custom_class(self, img):
        # Add a 4th dimension for batch size (as Keras expects)
        img_array = np.expand_dims(img, axis=0)

        cost = self.grab_cost_from_model([img_array, 0])[0]
        return cost * 100

    def get_class_id(self, class_name):
        return self.classname_to_id[class_name]
