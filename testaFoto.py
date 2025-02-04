from ultralytics import YOLO
from time import sleep

# Carregar o modelo treinado
model = YOLO("runs/detect/train8/weights/best.pt")

# Realizar a predição
results = model.predict("setupRodrigo.jpg", conf=0.3, show=True)
results[0].save()  # Salva a imagem com as caixas delimitadoras

# Acessar os resultados de detecção
for result in results:
    # Acessar as caixas de detecção
    boxes = result.boxes
    print(boxes.xywh)  # Coordenadas no formato xywh
    print(boxes.cls)   # Classes das detecções
    print(boxes.conf)  # Confiança das detecções
    sleep(3) #Deixa a imagem sendo mostrada na tela por 3 segundos
