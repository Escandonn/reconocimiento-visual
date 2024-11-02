import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

def get_images_with_id(base_path, image_size=(100, 100)):
    faces = []
    ids = []
    
    for person_id, person_name in enumerate(os.listdir(base_path)):
        person_path = os.path.join(base_path, person_name)
        
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if face_img is not None:
                    face_img = cv2.resize(face_img, image_size)
                    faces.append(face_img)
                    ids.append(person_id)

    return np.array(ids), faces

def train_model():
    path = 'dataSet'
    ids, faces = get_images_with_id(path)

    if len(faces) == 0 or len(ids) == 0:
        print("No se encontraron imágenes o etiquetas. Verifica la estructura del directorio.")
        return

    face_recognizer = cv2.face.FisherFaceRecognizer_create()
    face_recognizer.train(faces, ids)
    face_recognizer.save('modelo_fisherface.xml')
    print("Modelo entrenado y guardado como 'modelo_fisherface.xml'.")

def predict(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (100, 100))

    face_recognizer = cv2.face.FisherFaceRecognizer_create()
    face_recognizer.read('modelo_fisherface.xml')
    
    label, confidence = face_recognizer.predict(gray_img)

    threshold = 80.0
    if confidence < threshold:
        return label, confidence
    else:
        return -1, confidence  # -1 indica "Ninguno"

def upload_image():
    file_path = filedialog.askopenfilename()
    
    if file_path:
        test_img = cv2.imread(file_path)
        label, confidence = predict(test_img)

        # Asignar nombres a las etiquetas
        names = {0: "Alejandro Escandon", 1: "Juan Jordan", 2: "Kevin Hoy"}
        
        if label in names:
            result = f"Es {names[label]}"
        else:
            result = "Ninguno"
        
        messagebox.showinfo("Resultado", f"{result} con confianza {confidence:.2f}")

def recognize_video():
    cap = cv2.VideoCapture(0)
    
    match_count = 0
    total_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        label, confidence = predict(frame)

        # Asignar nombres a las etiquetas
        names = {0: "Alejandro Escandon", 1: "Juan Jordan", 2: "Kevin Hoy"}
        
        if label in names:  # Si hay una coincidencia
            total_count += 1
            match_count += 1
            cv2.putText(frame, f'{names[label]} Conf: {confidence:.2f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            total_count += 1
            cv2.putText(frame, 'Ninguno', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Mostrar el video en tiempo real
        cv2.imshow('Reconocimiento Facial', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Mostrar estadísticas
    if total_count > 0:
        messagebox.showinfo("Estadísticas", f"Coincidencias: {match_count}/{total_count}")

# Configurar ventana Tkinter
root = tk.Tk()
root.title("Reconocimiento Facial")
root.geometry("300x200")

# Botón para reentrenar modelo
train_btn = tk.Button(root, text="Reentrenar Modelo", command=train_model)
train_btn.pack(pady=10)

# Botón para subir imagen para predicción
upload_btn = tk.Button(root, text="Subir Imagen", command=upload_image)
upload_btn.pack(pady=10)

# Botón para reconocer en video
video_btn = tk.Button(root, text="Reconocer en Video", command=recognize_video)
video_btn.pack(pady=10)

root.mainloop()