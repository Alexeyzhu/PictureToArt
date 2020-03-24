from species import Species
import numpy as np


class Population(object):

    def __init__(self, pop_size, pop_cut, pic_size, poly_num, t_img):
        self.species = []
        self.population_size = pop_size
        self.pop_cut = pop_cut
        self.tensor_img = t_img
        self.image_size = pic_size
        self.number_of_polygons = poly_num
        for i in range(pop_size):
            self.species.append(Species(self.image_size,
                                        self.number_of_polygons,
                                        self.tensor_img))

    def get_fittest(self):
        self.species.sort(key=lambda x: x.fitness, reverse=True)
        return self.species[0]

    def mutative_crossover(self):
        size_of_population = len(self.species)
        descendants = []
        num_to_select = int(np.floor(size_of_population * self.pop_cut))
        num_of_random = int(np.ceil(1 / self.pop_cut)) - 1

        self.species.sort(key=lambda x: x.fitness, reverse=True)

        for i in range(num_to_select):
            for j in range(num_of_random):
                rand_parent = int(np.random.uniform(0, 1) * num_to_select)
                s = Species(self.image_size,
                            self.number_of_polygons,
                            self.tensor_img,
                            self.species[i].dna,
                            self.species[rand_parent].dna)
                descendants.append(s)

        self.species = self.species[0:num_to_select] + descendants
        self.species = self.species[0:size_of_population]
