import os
import random
import matplotlib.pyplot as plt
from rembg import remove
import cv2
import numpy as np
from tensorflow import Tensor, constant, reshape
from PIL import Image

random.seed(0)


def redimensionar_dataset(
    imagens_restantes, path="asl-alphabet/asl_alphabet_train/asl_alphabet_train/"
):
    for pasta in os.scandir(path):
        imagens = os.listdir(pasta)
        imagens_na_pasta = len(imagens)

        if imagens_na_pasta > imagens_restantes:
            num_files_to_delete = imagens_na_pasta - imagens_restantes

            for i in range(num_files_to_delete):
                imagem_deletar = pasta.path + "/" + random.choice(imagens)
                os.remove(imagem_deletar)
                imagens.remove(os.path.basename(imagem_deletar))
            print(f"Dataset redimensionado para {imagens_restantes} amostras por classe.")

        else:
            print(f"Dataset já possui {imagens_restantes} amostras por classe.")
            break


def processar_imagem(img: Tensor, label: Tensor):
    img = np.asarray(img).astype("float32")
    img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((200, 200)).convert("RGB"))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_no_back = remove(img_rgb)

    img_gray = cv2.cvtColor(img_no_back, cv2.COLOR_RGB2GRAY)

    alpha = 40 / (1.7 * img_gray.mean())
    img_equal = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=8)
    img_equal = cv2.GaussianBlur(img_equal, (5, 5), 0.9, 0.9)

    img_canny = cv2.Canny(img_equal, threshold1=90, threshold2=200)
    img_canny = img_canny / 255

    img_canny = constant(img_canny, shape=(200, 200))
    label = constant(label, shape=())
    return img_canny, label


def reajustar_imagem(img, label):
    img = reshape(img, shape=(32, 200, 200, 3))
    return img, label

def aplicar_todas_imagens(path):
    def preprocessar_imagem(imagem_cv2):
        img = cv2.imread(imagem_cv2)
        img_no_back = remove(img)
        img_gray = cv2.cvtColor(img_no_back, cv2.COLOR_RGB2GRAY)

        alpha = 40 / (1.7 * img_gray.mean())
        img_equal = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=8)
        img_equal = cv2.GaussianBlur(img_equal, (5, 5), 0.9, 0.9)
        img_canny = cv2.Canny(img_equal, threshold1=90, threshold2=200)
        img_canny = img_canny / 255

        plt.imsave(imagem_cv2, img_canny)

    for pasta in os.scandir(path):
        lista_imagens = os.listdir(pasta)
        for imagem in lista_imagens:
            caminho_imagem = pasta.path + "/" + imagem
            preprocessar_imagem(caminho_imagem)
