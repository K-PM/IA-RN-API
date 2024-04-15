from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import time  
from keras.models import load_model
import cv2
import numpy as np

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Cargar el modelo una vez al iniciar la aplicación
    modelo = load_model(os.path.abspath('modelo.h5'))
    @app.route('/guardar-imagen', methods=['POST'])
    def guardar_imagen():
        try:
            image_data = request.json['imageData']
            base64_data = image_data.split(',')[1]
            binary_data = base64.b64decode(base64_data)
            filename = f"captura_{int(time.time())}.jpeg"
            filepath = os.path.join('public', filename)
            with open(filepath, 'wb') as f:
                f.write(binary_data)

            predicted_class = predecir_clase_imagen(filepath)
            return jsonify({"message": "Clase predicha correctamente", "predictedClass": predicted_class}), 200

        except KeyError:
            return jsonify({"message": "No se proporcionó imageData en la solicitud"}), 400
        except Exception as e:
            print("Error:", e)
            return jsonify({"message": "Error al guardar la imagen"}), 500

    def predecir_clase_imagen(filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img = np.array(img).reshape(-1, 64, 64, 1)

        prediction = modelo.predict(img)

        # Obtener la clase predicha
        classes = ['aguja', 'alfiler', 'boton', 'cinta_metrica', 'descosedor', 'hilo_carrete', 'regla_L', 'tijera', 'tiza', 'trazador']
        predicted_class = classes[np.argmax(prediction)]

        print(f'La clase predicha es: {predicted_class}')
        return predicted_class
    return app

if __name__ == '__main__':
    app = create_app()
    app.run()
