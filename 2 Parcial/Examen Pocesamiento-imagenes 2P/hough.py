import cv2
import numpy as np

class DetectorHough:
    def __init__(self, ruta_imagen):
        self.imagen_original = cv2.imread(ruta_imagen)
        if self.imagen_original is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")

    def detectar_lineas(self):
        # 1. Escala de grises
        gris = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)

        # 2. Umbralización adaptativa para tolerar variaciones de luz
        binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 10)

        # 3. Operaciones morfológicas para limpiar ruido pequeño
        kernel = np.ones((3, 3), np.uint8)
        binaria_limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
        binaria_limpia = cv2.dilate(binaria_limpia, kernel, iterations=1)  # opcional

        # 4. Detección de bordes
        bordes = cv2.Canny(binaria_limpia, 20, 100)

        # 5. Detección de líneas
        lineas = cv2.HoughLines(bordes, 1, np.pi / 180, 73)

        # 6. Dibujar líneas sobre la imagen original
        resultado = self.imagen_original.copy()

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
                cv2.line(resultado, (x1, y1), (x2, y2), (0, 0, 255), 1)

        return resultado

    def mostrar_resultados(self):
        resultado = self.detectar_lineas()
        cv2.imshow("Líneas detectadas sobre imagen original", resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ruta = "Hough2.PNG"
    detector = DetectorHough(ruta)
    detector.mostrar_resultados()
