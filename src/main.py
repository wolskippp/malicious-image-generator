from src.FakeImgGenerator import FakeImgGenerator
from src.Utils import Utils


if __name__ == '__main__':
    img_name = "Maltese_dog"
    class_name_to_fake = "toaster"
    img_path = Utils.get_test_image_path(img_name)

    p_max = 0.5 # max 1
    p_min = -0.5 # min -1
    fake_class_prob_to_get = 0.8
    fake_img_generator = FakeImgGenerator(img_path,
                                          class_name_to_fake,
                                          p_max,
                                          p_min,
                                          fake_class_prob_to_get)

    population_size = 300
    pixels_percentage_to_change = 0.1
    max_generations_count = 1000
    population_percentage_to_keep = 0.5
    mutation_prob = 0.5
    crossover_prob = 0.5
    fake_img_path = fake_img_generator.run(population_size,
                                           pixels_percentage_to_change,
                                           max_generations_count,
                                           population_percentage_to_keep,
                                           mutation_prob,
                                           crossover_prob)

