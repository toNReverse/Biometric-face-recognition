from ultralytics import YOLO
import cv2

# === PARAMETRI ===
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Modello addestrato
VIDEO_PATH = "video.mp4"  # Sostituisci con il tuo file video

# === CARICAMENTO MODELLO ===
model = YOLO(MODEL_PATH)

# Verifica delle classi
print("Classi del modello:", model.names)

# === CARICAMENTO VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Errore nell'aprire il video: {VIDEO_PATH}")
    exit()

print("Analisi del video in corso. Premi ESC per interrompere.")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Fine del video

    # Rilevamento
    results = model(frame, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Mostra il risultato
    cv2.imshow("Riconoscimento Facciale da Video", frame)

    # ESC per uscire
    if cv2.waitKey(1) == 27:
        print("Analisi interrotta.")
        break

cap.release()
cv2.destroyAllWindows()
