from pathlib import Path

import cv2
import matplotlib.pyplot as plt


IMAGE_FILENAME = "WhatsApp Image 2026-11-24 at 17.46.14.jpeg"


def main():
    image_path = Path(__file__).with_name(IMAGE_FILENAME)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    new_width = 640
    new_height = max(1, round(height * new_width / width))
    resized = cv2.resize(
        image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    print("Dimensões originais:", image_rgb.shape)
    print("Dimensões após redimensionar:", resized.shape)

    x1, x2 = int(0.30 * width), int(0.88 * width)
    y1, y2 = int(0.35 * height), int(0.83 * height)
    crop = image_rgb[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    print("Dimensões do recorte:", crop.shape)

    blur5 = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    blur21 = cv2.GaussianBlur(image_rgb, (21, 21), 0)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(smooth, 100, 200)

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
    main()
