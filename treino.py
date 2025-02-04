from ultralytics import YOLO
import torch

print(torch.cuda.is_available())  # Deve retornar True
print(torch.cuda.device_count())  # Número de GPUs disponíveis
print(torch.cuda.get_device_name(0))  # Nome da GPU

if __name__ == '__main__':
    model = YOLO("yolov8m.pt")  # Modelo pré-treinado 
    model.to("cuda") # Move o modelo para a GPU para acelerar o processamento
    model.train(data="data.yaml", epochs=50, imgsz=640, batch=16, device="cuda") # Treina o modelo usando os dados definidos em 'data.yaml'

    model.val() # Executa a validação do modelo

    model.predict("setupRodrigo.jpg", conf=0.3, show=True) # Faz uma previsão em uma imagem específica