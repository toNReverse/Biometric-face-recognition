from ultralytics import YOLO
import cv2

# === PARAMETRI ===
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Il tuo modello addestrato
IMAGE_PATH = "img.jpeg"  # Sostituisci con il percorso della tua immagine

# === CARICAMENTO MODELLO ===
model = YOLO(MODEL_PATH)

# Verifica delle classi
print("Classi del modello:", model.names)

# === CARICAMENTO IMMAGINE ===
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Errore nel caricamento dell'immagine da: {IMAGE_PATH}")
    exit()

# Rilevamento
results = model(image, verbose=False)

# Disegna i risultati
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    name = model.names[cls_id]  # nome della classe

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

# Mostra il risultato
cv2.imshow("Riconoscimento Facciale da Immagine", image)
cv2.waitKey(0)
cv2.destroyAllWindows()