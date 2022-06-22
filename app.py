import numpy as np
import tensorflow as tf

from flask import Flask, request
from config import config

# Cargar el modelo pre entrenado
with open('santanderImage_model.json', 'r') as f:
    model_json = f.read()
model = tf.keras.models.model_from_json(model_json)

# Cargar los pesos en el modelo
model.load_weights("santanderImage_model.h5")


# Método para predecir
def predict(image_url):
    # Predecir nuevos datos
    class_names = ['otro', 'santander']
    #image_url = "https://becasmexico.org/wp-content/uploads/2020/10/detectada-campana-fraudulenta-de-phishing-al-banco-santander.png"
    image_path = tf.keras.utils.get_file(origin=image_url)

    img = tf.keras.utils.load_img(
        image_path, target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    porcentSimil = 100 * np.max(score)
    className = class_names[np.argmax(score)]

    return {'className': className, 'porcentSimil': porcentSimil}

# app Flask
app = Flask(__name__)

@app.route('/image_class', methods = ['POST'])
def prueba():
    try:
        requestPredict = request.json
        print(requestPredict['image_url'])
        response = predict(requestPredict['image_url'])
        return response
    except Exception as ex:
        return ex

def pagina_no_encontrada(error):
    return "<h1>La página a la que intentas acceder no existe....</h1>", 404


if __name__ == '__main__':
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(port=5000)
    app.run(host='0.0.0.0', debug=True)