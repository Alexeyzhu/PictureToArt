import cv2
import torch
from tqdm import tqdm
from population import Population

folder = "image.jpg"
pic_size = 64
drawing_pic_size = 512

population_size = 30
number_of_polygons = 50
pop_cut = 0.1
iterations = 4000
output_pic_name = "output_image.png"


def pic_show(img, img_name=output_pic_name):
    cv2.imshow(img_name, img)
    cv2.imwrite(img_name, img=img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def prepare_image(img_size):
    img = cv2.imread(folder)

    height, width, _ = img.shape

    if abs(height - width) > 2:
        cut = (max(height, width) - min(height, width)) // 2
        if height > width:
            img = img[cut: -cut]
        elif width > height:
            img = img[:, cut: -cut]

    img = cv2.resize(img, (img_size, img_size))
    return img


def main():
    original = prepare_image(pic_size)

    tensor_img = torch.from_numpy(original).float().cuda()
    full_tensor = torch.from_numpy(prepare_image(drawing_pic_size)).float().cuda()

    res_img = torch.zeros_like(tensor_img).float().cuda()

    population = Population(population_size, pop_cut, pic_size, number_of_polygons, tensor_img)

    for i in tqdm(range(iterations)):
        population.mutative_crossover()
        best = population.get_fittest()
        res_img = best.polygons_to_canvas(full_tensor, drawing_pic_size)
        if i % 10 == 0:
            cv2.imwrite("s_" + output_pic_name, img=torch.round(res_img).byte().cpu().numpy())

    pic_show(torch.round(res_img).byte().cpu().numpy())


main()
