from keras_classifier import Keras

if __name__ == '__main__':
    image_to_test = "cat1"
    # Will tell you what this really is
    Keras.get_prediction(image_to_test)

    # Choose an ImageNet object to fake
    # The list of classes is available here: https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b
    # Class #859 is "toaster"
    Keras.check_prediction_on_custom_class(image_to_test, 859)
