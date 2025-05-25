from ultralytics import YOLO
import cv2

# === PARAMETRI ===
MODEL_PATH = "runs/detect/train/weights/best.pt"  # il tuo modello addestrato

# === CARICAMENTO MODELLO ===
model = YOLO(MODEL_PATH)

# Verifica delle classi
print("Classi del modello:", model.names)

# === AVVIO DELLA VIDEOCAMERA ===
cap = cv2.VideoCapture(0)

print("Riconoscimento facciale avviato. Premi ESC per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rilevamento
    results = model(frame, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]  # nome della classe

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Mostra il risultato
    cv2.imshow("Riconoscimento Facciale", frame)

    # Premi ESC per uscire
    if cv2.waitKey(1) == 27:
        print("Riconoscimento facciale terminato.")
        break

# Rilascia la webcam e chiudi le finestre
cap.release()
cv2.destroyAllWindows()