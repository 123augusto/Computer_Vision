from pathlib import Path

import matplotlib.pyplot as plt

from imagem import Imagem


IMAGE_FILENAME = "WhatsApp Image 2026-11-24 at 17.46.14.jpeg"


class Main:
    def __init__(self, imagem: Imagem):
        self.imagem = imagem

    def executar(self):
        image_rgb = self.imagem.rgb
        height, width, channel_count = image_rgb.shape

        print("Altura:", height)
        print("Largura:", width)
        print("Canais:", channel_count)
        print("Tipo de dados:", image_rgb.dtype)

        x, y = 100, 50
        if 0 <= x < width and 0 <= y < height:
            print(f"Pixel RGB em ({x}, {y}):", image_rgb[y, x])
        else:
            print(f"O pixel ({x}, {y}) está fora da imagem de {width} × {height} pixels.")

        gray = self.imagem.tons_de_cinza()
        resized = self.imagem.redimensionar(640)
        crop = self.imagem.recortar(0.30, 0.35, 0.88, 0.83)
        blur5 = self.imagem.desfocar(5)
        blur21 = self.imagem.desfocar(21)
        edges = self.imagem.detectar_bordas(gray, 100, 200)

        print("Dimensões originais:", image_rgb.shape)
        print("Dimensões após redimensionar:", resized.shape)
        print("Dimensões do recorte:", crop.shape)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for index, (ax, name) in enumerate(zip(axes, ("Vermelho", "Verde", "Azul"))):
            ax.imshow(image_rgb[:, :, index], cmap="gray", vmin=0, vmax=255)
            ax.set_title(name)
            ax.axis("off")
        fig.tight_layout()

        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        panels = [
            (image_rgb, "Imagem original"),
            (gray, "Tons de cinza"),
            (resized, "Imagem redimensionada"),
            (crop, "Região recortada do carro"),
            (blur5, "Gaussiano 5 × 5"),
            (blur21, "Gaussiano 21 × 21"),
        ]
        for ax, (image, title) in zip(axes.flat, panels):
            if image.ndim == 2:
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(image)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(edges, cmap="gray", vmin=0, vmax=255)
        ax.set_title("Bordas detectadas por Canny")
        ax.axis("off")
        fig.tight_layout()

        plt.show()


if __name__ == "__main__":
    caminho = Path(__file__).with_name(IMAGE_FILENAME)
    Main(Imagem(caminho)).executar()
