import time
from src.Utils import Utils
from datetime import timedelta
import matplotlib.pyplot as plt

class Result(object):
    def __init__(self,
                 class_name_to_fake,
                 p_max,
                 p_min,
                 fake_class_prob_to_get,
                 population_size,
                 pixels_percentage_to_change,
                 max_generation_count,
                 population_percentage_to_keep,
                 mutation_prob,
                 crossover_prob):
        self.class_name_to_fake = class_name_to_fake
        self.p_max = p_max
        self.p_min = p_min
        self.fake_class_prob_to_get = fake_class_prob_to_get
        self.population_size = population_size
        self.pixels_percentage_to_change = pixels_percentage_to_change
        self.max_generation_count = max_generation_count
        self.population_percentage_to_keep = population_percentage_to_keep
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

        self.start_time = None
        self.running_time = None
        self.best_img = None
        self.results = []

    def start(self):
        self.start_time = time.time()

    def add_best_current_probability(self, best_probability):
        self.results.append(best_probability)

    def set_result_img(self, img):
        self.best_img = img

    def stop(self):
        self.running_time = time.time() - self.start_time

    def get_max_probability(self):
        return max(self.results)

    def save(self, output_dir):
        Utils.save_chart(self._generate_chart(), Utils.get_path("chart.png", basepath=output_dir))

        summary = self._generate_summary()
        summary_file_path = Utils.get_path("summary.txt", basepath=output_dir)
        with open(summary_file_path, 'w') as outfile:
            outfile.write(summary)

        result_img_file_path = Utils.get_path("result_img.jpg", basepath=output_dir)
        Utils.save_img(img=self.best_img, img_path=result_img_file_path)

    def _generate_chart(self):
        x_data = range(1, len(self.results)+1)
        y_data = self.results
        plt.scatter(x_data, y_data)
        plt.xticks(x_data)
        plt.ylabel("Fake class probability")
        plt.xlabel("Generations")
        plt.title("Results in following generations")
        return plt

    def _get_formatted_running_time(self):
        return str(timedelta(seconds=self.running_time))

    def _generate_summary(self):
        return "\n".join(["### Configuration:",
                         "Class to fake: {}".format(self.class_name_to_fake),
                         "P_max: {}".format(self.p_max),
                         "P_min: {}".format(self.p_min),
                         "Fake class probability to get: {}".format(self.fake_class_prob_to_get),
                         "Population size: {}".format(self.population_size),
                         "Pixels percentage to change: {}".format(self.pixels_percentage_to_change),
                         "Max generations count: {}".format(self.max_generation_count),
                         "Population percentage to keep".format(self.population_percentage_to_keep),
                         "Mutation probability: {}".format(self.mutation_prob),
                         "Crossover probability: {}".format(self.crossover_prob),
                         "",
                         "### Results:",
                         "Running time: {}".format(self._get_formatted_running_time())])
