import cv2
import os

letra = input("Digite a letra que você vai capturar (ex: A): ").upper()
save_dir = f"dataset/{letra}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Região onde a mão deve ser posicionada
    roi = frame[100:300, 100:300]

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f"Capturando: {letra} - {img_count} imagens", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("Captura", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):  # Pressione 'c' para capturar uma imagem
        img_path = os.path.join(save_dir, f"{img_count}.jpg")
        cv2.imwrite(img_path, roi)
        img_count += 1

    elif key == ord('q'):  # Pressione 'q' para sair
        break

cap.release()
cv2.destroyAllWindows()