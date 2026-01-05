import cv2
import numpy as np

class DetectorHough:
    def __init__(self, ruta_imagen):
        self.imagen_original = cv2.imread(ruta_imagen)
        if self.imagen_original is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        self.imagen_gris = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        _, self.imagen_binaria = cv2.threshold(self.imagen_gris, 240, 255, cv2.THRESH_BINARY_INV)
        self.bordes = cv2.Canny(self.imagen_binaria, 30, 150, apertureSize=3)

    def detectar_lineas(self):
        copia = self.imagen_original.copy()
        lineas = cv2.HoughLines(self.bordes, 1, np.pi / 180, 100)
        if lineas is not None:
            for linea in lineas:
                rho, theta = linea[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(copia, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return copia

    def detectar_circulos(self):
        copia = self.imagen_original.copy()
        circulos = cv2.HoughCircles(
            self.imagen_gris,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=150,
            param2=60,
            minRadius=15,
            maxRadius=60
        )
        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for c in circulos[0, :]:
                cv2.circle(copia, (c[0], c[1]), c[2], (0, 255, 0), 2)
                cv2.circle(copia, (c[0], c[1]), 2, (255, 0, 0), 3)
        return copia

    def mostrar_resultados(self):
        cv2.imshow("Imagen original", self.imagen_original)
        imagen_lineas = self.detectar_lineas()
        cv2.imshow("Líneas detectadas", imagen_lineas)
        imagen_circulos = self.detectar_circulos()
        cv2.imshow("Círculos detectados", imagen_circulos)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ruta = "carro.jpg"
    detector = DetectorHough(ruta)
    detector.mostrar_resultados()
