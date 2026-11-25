from pathlib import Path

import cv2


class Imagem:
    def __init__(self, caminho: Path):
        bgr = cv2.imread(str(caminho))
        if bgr is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho}")

        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def tons_de_cinza(self):
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)

    def redimensionar(self, nova_largura):
        altura, largura = self.rgb.shape[:2]
        nova_altura = max(1, round(altura * nova_largura / largura))
        return cv2.resize(
            self.rgb, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA
        )

    def recortar(self, esquerda, topo, direita, base):
        altura, largura = self.rgb.shape[:2]
        x1, x2 = int(esquerda * largura), int(direita * largura)
        y1, y2 = int(topo * altura), int(base * altura)
        return self.rgb[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)].copy()

    def desfocar(self, tamanho):
        return cv2.GaussianBlur(self.rgb, (tamanho, tamanho), 0)

    def detectar_bordas(self, cinza, limiar_inferior, limiar_superior):
        suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
        return cv2.Canny(suavizada, limiar_inferior, limiar_superior)
