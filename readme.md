# AUTORES

## Alejandro Escandon
---
## Juan David Jordan
---
## Kevin Koy
---

---

# Tutorial de Reconocimiento Facial con OpenCV y Tkinter

## Introducción

Este proyecto utiliza la biblioteca OpenCV para implementar un sistema de reconocimiento facial que puede identificar a tres personas (Alejandro Escandon, Juan David Jordan y Kevin Koy) a partir de imágenes estáticas o video en tiempo real. La interfaz gráfica se gestiona mediante Tkinter, lo que permite al usuario interactuar fácilmente con el sistema.

## Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas:

- Python 3.x
- OpenCV
- NumPy
- Tkinter (generalmente incluido con Python)

Puedes instalar OpenCV y NumPy usando pip:

```bash
pip install opencv-python opencv-contrib-python numpy
```

## Estructura del Proyecto

La estructura del proyecto debe ser la siguiente:

```
/tu_proyecto
│
├── dataSet/
│   ├── AlejandroEscandon/
│   │   ├── imagen1.jpg
│   │   ├── imagen2.jpg
│   │   └── ...
│   ├── JuanDavidJordan/
│   │   ├── imagen1.jpg
│   │   ├── imagen2.jpg
│   │   └── ...
│   └── KevinKoy/
│       ├── imagen1.jpg
│       ├── imagen2.jpg
│       └── ...
├── main.py
└── README.md
```

- **dataSet/**: Carpeta que contiene subcarpetas para cada persona, donde se almacenan las imágenes de entrenamiento.
- **main.py**: Archivo principal que contiene el código del sistema de reconocimiento facial.
- **README.md**: Este archivo.

## Descripción del Código

### 1. Importación de Bibliotecas

```python
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
```

Se importan las bibliotecas necesarias:
- `os`: Para manejar operaciones del sistema de archivos.
- `cv2`: Para el procesamiento de imágenes y video.
- `numpy`: Para manejar arreglos numéricos.
- `tkinter`: Para crear la interfaz gráfica.

### 2. Función `get_images_with_id`

```python
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
```

Esta función carga las imágenes desde el directorio especificado (`base_path`), redimensiona cada imagen y asigna un ID a cada persona basado en su posición en el directorio.

### 3. Función `train_model`

```python
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
```

Esta función entrena el modelo de reconocimiento facial utilizando las imágenes cargadas. El modelo entrenado se guarda como un archivo XML para su uso posterior.

### 4. Función `predict`

```python
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
```

Esta función toma una imagen de prueba (`test_img`), la convierte a escala de grises y la redimensiona. Luego utiliza el modelo entrenado para predecir si la imagen corresponde a Alejandro, Juan o Kevin.

### 5. Función `upload_image`

```python
def upload_image():
    file_path = filedialog.askopenfilename()
    
    if file_path:
        test_img = cv2.imread(file_path)
        label, confidence = predict(test_img)

        if label == 0:
            result = "Es Alejandro Escandon"
        elif label == 1:
            result = "Es Juan David Jordan"
        elif label == 2:
            result = "Es Kevin Koy"
        else:
            result = "Ninguno"
        
        messagebox.showinfo("Resultado", f"{result} con confianza {confidence:.2f}")
```

Esta función permite al usuario seleccionar una imagen desde su sistema. Luego realiza la predicción y muestra el resultado en un cuadro de mensaje.

### 6. Función `recognize_video`

```python
def recognize_video():
    cap = cv2.VideoCapture(0)
    
    match_count = 0
    total_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        label, confidence = predict(frame)

        if label != -1:  # Si hay una coincidencia
            total_count += 1
            match_count += 1
            cv2.putText(frame, f'ID: {label} Conf: {confidence:.2f}', (10, 30), 
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
```

Esta función captura video en tiempo real desde la cámara y aplica el reconocimiento facial a cada fotograma. Muestra estadísticas sobre cuántas coincidencias se encontraron.

### 7. Configuración de la Interfaz Gráfica

```python
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
```

Aquí se configura la ventana principal de Tkinter con botones para reentrenar el modelo, subir una imagen y reconocer en video.

## Ejecución del Programa

Para ejecutar el programa:

1. Asegúrate de que tu estructura de carpetas esté configurada correctamente.
2. Coloca imágenes de Alejandro, Juan y Kevin en sus respectivas carpetas dentro de `dataSet`.
3. Ejecuta el archivo `main.py`:

```bash
python main.py
```

4. Usa los botones en la interfaz gráfica para reentrenar el modelo o realizar predicciones.

## Conclusiones

Este proyecto proporciona una base sólida para implementar un sistema básico de reconocimiento facial utilizando OpenCV y Tkinter. Puedes expandirlo añadiendo más características como almacenamiento en bases de datos o integración con otras aplicaciones.