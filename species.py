import numpy as np
import torch
import cv2
from random import random


class Species(object):
    def __init__(self,
                 pic_size,
                 poly_n,
                 t_img,
                 parent1=None, parent2=None):
        self.dna = []
        self.fitness = 0
        self.pic_size = pic_size
        self.chance_to_mutate = 0.01
        self.amount_of_mutation = 0.1

        if parent1 is not None and parent2 is not None:
            dna_size = len(parent1)
            turn, _ = np.random.uniform(low=0,
                                        high=1,
                                        size=(2, dna_size))
            for i in range(0, dna_size, 10):
                if turn[i] < 0.5:
                    inherited_attribute = parent1
                else:
                    inherited_attribute = parent2

                for j in range(10):
                    mut = inherited_attribute[i + j]

                    if random() < self.chance_to_mutate:
                        mut += random() * self.amount_of_mutation + random() * self.amount_of_mutation - self.amount_of_mutation
                    if mut < 0:
                        mut = 0
                    elif mut > 1:
                        mut = 1

                    self.dna.append(mut)
        else:
            x0, y0, x1, y1, x2, y2 = np.random.uniform(low=0,
                                                       high=1,
                                                       size=(6, poly_n))
            red, green, blue = np.random.uniform(low=0,
                                                 high=1,
                                                 size=(3, poly_n))

            alpha, _ = np.random.uniform(low=0,
                                         high=1,
                                         size=(2, poly_n))

            for i in range(poly_n):
                self.dna.append(red[i])
                self.dna.append(green[i])
                self.dna.append(blue[i])
                self.dna.append(max(alpha[i], 0.2))
                self.dna.append(x0[i])
                self.dna.append(y0[i])
                self.dna.append(x1[i])
                self.dna.append(y1[i])
                self.dna.append(x2[i])
                self.dna.append(y2[i])

        canvas = self.polygons_to_canvas(t_img)
        self.fitness = self.fitness_function(tensor_img=t_img, canvas=canvas, pic_size=pic_size)

    def fitness_function(self, tensor_img, canvas, pic_size):
        return 1 - (((tensor_img - canvas) ** 2).sum().item() / (pic_size * pic_size * 3 * 256 * 256))

    def polygons_to_canvas(self, tensor_img, pic_size=None):
        canvas = torch.zeros_like(tensor_img).float().cuda()
        for i in range(0, len(self.dna), 10):
            if pic_size is None:
                new_img = np.zeros((self.pic_size, self.pic_size, 3))
                pts = np.array([[int(self.dna[i + 4] * self.pic_size), int(self.dna[i + 5] * self.pic_size)],
                                [int(self.dna[i + 6] * self.pic_size), int(self.dna[i + 7] * self.pic_size)],
                                [int(self.dna[i + 8] * self.pic_size), int(self.dna[i + 9] * self.pic_size)]])
            else:
                new_img = np.zeros((pic_size, pic_size, 3))
                pts = np.array([[int(self.dna[i + 4] * pic_size), int(self.dna[i + 5] * pic_size)],
                                [int(self.dna[i + 6] * pic_size), int(self.dna[i + 7] * pic_size)],
                                [int(self.dna[i + 8] * pic_size), int(self.dna[i + 9] * pic_size)]])
            alpha = self.dna[i + 3]
            pts = pts.reshape((-1, 1, 2))
            color = ([int(self.dna[i] * 255),
                      int(self.dna[i + 1] * 255),
                      int(self.dna[i + 2] * 255)])
            cv2.fillConvexPoly(img=new_img,
                               points=pts,
                               color=color)
            new_img = torch.from_numpy(new_img).float().cuda()
            new_img = torch.where(new_img == 0, canvas, (canvas * alpha + new_img * (1 - alpha)))
            canvas = new_img
        return canvas
