import itertools
import numpy as np

from src.FakeImgGenerator import FakeImgGenerator
from src.Utils import Utils
from Config import *
import matplotlib.pyplot as plt


def run_analysis():
    population_size = np.arange(start=10, stop=11, step=5)
    pixels_percentage_to_change = np.arange(start=0.1, stop=0.2, step=0.1)
    max_generation_count = [100]
    population_percentage_to_keep = np.arange(start=0.1,stop=0.2, step=1)
    mutation_prob = np.arange(start=0.1, stop=0.2, step=0.1)
    crossover_prob = np.arange(start=0.1, stop=0.2, step=0.1)

    all_params_combination = itertools.product(population_size,
                                               pixels_percentage_to_change,
                                               max_generation_count,
                                               population_percentage_to_keep,
                                               mutation_prob,
                                               crossover_prob)

    fake_img_generator = FakeImgGenerator(Utils.get_test_image_path(IMG_NAME),
                                          CLASS_NAME_TO_FAKE,
                                          P_MAX,
                                          P_MIN,
                                          FAKE_CLASS_PROB_TO_GET)

    results = []
    output_dir = Utils.get_new_output_dir()
    for i, c in enumerate(all_params_combination):
        generation_result = fake_img_generator.run(population_size=c[0],
                                                   pixels_percentage_to_change=c[1],
                                                   max_generations_count=c[2],
                                                   population_percentage_to_keep=c[3],
                                                   mutation_prob=c[4],
                                                   crossover_prob=c[5])
        results.append(generation_result)
        result_path = Utils.create_child_dir(basepath=output_dir, child_dir="result_{}".format(str(i)))
        generation_result.save(output_dir=result_path)

    _save_results_comparison(results, output_dir)


def _save_results_comparison(results, output_dir):
    def generate_comparison_summary_file():
        content = []
        results_probabilities_times = [(r.get_max_probability(), r.running_time) for r in results]

        if all([prob_time[0] < FAKE_CLASS_PROB_TO_GET for prob_time in results_probabilities_times]):
            content.append("No parameters set exceeded FAKE_CLASS_PROB_TO_GET ({})\n".format(FAKE_CLASS_PROB_TO_GET))
            best_result = max([prob_time[0] for prob_time in results_probabilities_times])
            for i, prob_time in enumerate(results_probabilities_times):
                if prob_time[0] == best_result:
                    best_result_id = i
                    best_result_time = prob_time[1]
            content.append("Best probability result is {}, received for paramaters set number {}\n"
                           " in time {}.".format(best_result, best_result_id, best_result_time))
        else:
            prob, shortest_time = min([prob_time for prob_time in results_probabilities_times
                                    if prob_time[0] >= FAKE_CLASS_PROB_TO_GET], key= lambda prob_time : prob_time[1])
            for i, prob_time in enumerate(results_probabilities_times):
                if prob_time[0] == prob and prob_time[1] == shortest_time:
                    best_result_id = i
            content.append("FAKE_CLASS_PROB_TO_GET ({}) was exceeded in the shortest time ({}s) "
                           "by parameters set number {} and is {}".format(FAKE_CLASS_PROB_TO_GET,
                                                                          shortest_time, best_result_id, prob))

        return "".join(content)

    def get_results_timings_chart():
        x_data = [r.get_max_probability() for r in results]
        y_data = [r.running_time for r in results]
        fig, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        for i, prob in enumerate(x_data):
            ax.annotate(i, (prob, y_data[i]))

        plt.ylabel("Time [s]")
        plt.xlabel("Probability")
        plt.title("Results and times")
        return plt

    def get_results_chart():
        x_data = range(len(results))
        y_data = [r.get_max_probability() for r in results]
        plt.scatter(x_data, y_data)
        plt.xticks(x_data)
        plt.ylabel("Probability")
        plt.xlabel("Parameters set ID")
        plt.title("Results in following parameters sets")
        return plt

    def get_timings_chart():
        x_data = range(len(results))
        y_data = [r.running_time for r in results]
        plt.scatter(x_data, y_data)
        plt.xticks(x_data)
        plt.ylabel("Time [s]")
        plt.xlabel("Parameters set ID")
        plt.title("Running times in following parameters sets")
        return plt

    Utils.save_chart(get_results_chart(), Utils.get_path("results_chart.png", basepath=output_dir))
    Utils.save_chart(get_timings_chart(), Utils.get_path("timings_chart.png", basepath=output_dir))
    Utils.save_chart(get_results_timings_chart(), Utils.get_path("timings_results_chart.png", basepath=output_dir))

    summary_file = generate_comparison_summary_file()
    summary_file_path = Utils.get_path("summary.txt", basepath=output_dir)
    with open(summary_file_path, 'w') as outfile:
        outfile.write(summary_file)


run_analysis()