import numpy as np
import cv2
import tensorflow as tf

# Yüz tespiti için Haar Cascade yükleniyor
face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Kamera ayarları
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
settings = {
    'scaleFactor': 1.3,
    'minNeighbors': 5,
    'minSize': (50, 50)
}

labels = ["sinirli","igrenmis","korkmus","mutlu","notr","uzgun","saskin"]

# Eğitilmiş modelin yüklenmesi
model = tf.keras.models.load_model('my_model.keras')
last_state = None

try:
    while True:
        ret, img = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = face_detection.detectMultiScale(gray, **settings)

        for x, y, w, h in detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (245, 135, 66), 2)
            cv2.rectangle(img, (x, y), (x + w // 3, y + 20), (245, 135, 66), -1)
            face = gray[y + 5:y + h - 5, x + 20:x + w - 20]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0

            predictions = model.predict(np.array([face.reshape((48, 48, 1))])).argmax()
            state = labels[predictions]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, state, (x + 10, y + 15), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)  

        cv2.imshow('Duygu Tespit', img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()