import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Caminhos
modelo_path = "modelo/modelo_libras.h5"
dataset_path = "dataset"

# Carregar modelo treinado
model = load_model(modelo_path)

# Obter labels diretamente das pastas do dataset (ordem alfabética)
labels = sorted(os.listdir(dataset_path))  # Ex: ['A', 'B', 'C', ..., 'Y'] (sem H, J, K, X, Z)

# Parâmetros da imagem
IMG_SIZE = 200

# Variável para armazenar o nome
nome_detectado = ""

# Iniciar webcam
cap = cv2.VideoCapture(0)

print("🔤 Pressione ESPAÇO para adicionar letra, BACKSPACE para remover, Q para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Região de interesse (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # Previsão
    pred = model.predict(roi_input, verbose=0)
    index = np.argmax(pred)
    predicted_label = labels[index]
    confidence = np.max(pred)

    # Mostrar informações na tela
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Letra: {predicted_label} ({confidence*100:.2f}%)", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Nome: {nome_detectado}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)

    cv2.imshow("Reconhecimento de Libras", frame)

    key = cv2.waitKey(1) & 0xFF

    # Adicionar letra ao nome
    if key == ord(' '):  # Tecla ESPAÇO
        nome_detectado += predicted_label
    elif key == 8:  # Tecla BACKSPACE
        nome_detectado = nome_detectado[:-1]
    elif key == ord('q'):  # Tecla Q para sair
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Nome final reconhecido: {nome_detectado}")
