import torch
from flask import Flask, render_template, request, jsonify
from modelo import CustomDenseNet, procesar_imagen, predecir_neumonia
import gdown

app = Flask(__name__)


# URL del archivo en Google Drive
url = 'https://drive.google.com/uc?id=1Ed9g2Rj_k7CPF8ClBalaYfDhfbNlsuTC'

# Descargar el archivo y guardarlo localmente
output = 'mejor_modelo.pth'
gdown.download(url, output, quiet=False)

# Cargar el modelo y moverlo al dispositivo adecuado (CPU o GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = CustomDenseNet(num_classes=2)
modelo.load_state_dict(torch.load(output, map_location=device))
modelo.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    try:
        imagen_tensor = procesar_imagen(file)
        prediccion = predecir_neumonia(modelo, imagen_tensor)
        if prediccion == 1:
            resultado = "La imagen muestra signos de neumonía."
        else:
            resultado = "La imagen no muestra signos de neumonía."
        return jsonify({"resultado": resultado}), 200
        return render_template('index.html', resultado=resultado)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
