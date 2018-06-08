from src.FakeImgGenerator import FakeImgGenerator
from src.Utils import Utils
from Config import *

if __name__ == '__main__':
    img_name = IMG_NAME
    img_path = Utils.get_test_image_path(img_name)

    p_max = P_MAX
    p_min = P_MIN
    fake_class_prob_to_get = FAKE_CLASS_PROB_TO_GET
    fake_img_generator = FakeImgGenerator(img_path,
                                          p_max,
                                          p_min,
                                          fake_class_prob_to_get
                                          )

    population_size = 30
    pixels_percentage_to_change = 0.4
    max_generations_count = 100
    population_percentage_to_keep = 0.1
    mutation_prob = 0.6
    crossover_prob = 0.7
    result = fake_img_generator.run(population_size,
                                    pixels_percentage_to_change,
                                    max_generations_count,
                                    population_percentage_to_keep,
                                    mutation_prob,
                                    crossover_prob)
    output_dir = Utils.get_new_output_dir()
    result.save(output_dir)

