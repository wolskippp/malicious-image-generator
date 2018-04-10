import numpy as np
from keras import backend as K
from keras.applications import inception_v3
from keras.preprocessing import image

from utils import Utils


class Keras:
    @staticmethod
    def get_prediction(test_image_name):
        # Load pre-trained image recognition model
        model = inception_v3.InceptionV3()

        full_path = Utils.get_test_image(test_image_name)
        # Load the image file and convert it to a numpy array
        img = image.load_img(full_path, target_size=(299, 299))
        input_image = image.img_to_array(img)

        # Scale the image so all pixel intensities are between [-1, 1] as the model expects
        input_image /= 255.
        input_image -= 0.5
        input_image *= 2.

        # Add a 4th dimension for batch size (as Keras expects)
        input_image = np.expand_dims(input_image, axis=0)

        # Run the image through the neural network
        predictions = model.predict(input_image)

        # Convert the predictions into text and print them
        predicted_classes = inception_v3.decode_predictions(predictions, top=1)
        imagenet_id, name, confidence = predicted_classes[0][0]
        print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))

    @staticmethod
    def check_prediction_on_custom_class(file_name, object_type_to_fake):
        # Load pre-trained image recognition model
        model = inception_v3.InceptionV3()

        # Grab a reference to the first and last layer of the neural net
        model_input_layer = model.layers[0].input
        model_output_layer = model.layers[-1].output

        full_path = Utils.get_test_image(file_name)
        # Load the image to hack
        img = image.load_img(full_path, target_size=(299, 299))
        original_image = image.img_to_array(img)

        # Scale the image so all pixel intensities are between [-1, 1] as the model expects
        original_image /= 255.
        original_image -= 0.5
        original_image *= 2.

        # Add a 4th dimension for batch size (as Keras expects)
        original_image = np.expand_dims(original_image, axis=0)
        # Create a copy of the input image to hack on
        hacked_image = np.copy(original_image)

        # Define the cost function.
        # Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
        cost_function = model_output_layer[0, object_type_to_fake]

        # We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
        # In this case, referring to "model_input_layer" will give us back image we are hacking.
        gradient_function = K.gradients(cost_function, model_input_layer)[0]

        # Create a Keras function that we can call to calculate the current cost and gradient
        grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                        [cost_function, gradient_function])

        cost = 0.0
        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

        print("Model's predicted likelihood that the image is a toaster: {:.8}%".format(cost * 100))
