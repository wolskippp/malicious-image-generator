from src.FakeImgGenerator import FakeImgGenerator
from src.Utils import Utils


if __name__ == '__main__':
    img_name = "toaster"
    # class_name_to_fake = "toy_poodle"
    img_path = Utils.get_test_image_path(img_name)

    p_max = 0.1 # max 1
    p_min = -0.1 # min -1
    fake_class_prob_to_get = 50
    fake_img_generator = FakeImgGenerator(img_path,
                                          p_max,
                                          p_min,
                                          fake_class_prob_to_get
                                          )

    population_size = 30
    pixels_percentage_to_change = 0.5
    max_generations_count = 1000
    population_percentage_to_keep = 0.2
    mutation_prob = 0.5
    crossover_prob = 0.7
    fake_img_path = fake_img_generator.run(population_size,
                                           pixels_percentage_to_change,
                                           max_generations_count,
                                           population_percentage_to_keep,
                                           mutation_prob,
                                           crossover_prob)

