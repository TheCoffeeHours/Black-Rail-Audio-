from flask import Flask, request, render_template, redirect, url_for, flash
import tensorflow as tf
import tensorflow_io as tfio
from audio_pipeline import detect_black_rail_calls 

app = Flask(__name__)

# Load the model (assuming it's saved as 'model.h5' in the current directory)
model = tf.keras.models.load_model('c:\\Users\\chris\\Desktop\\Black_Rail_Audio_Detection\\models\\BLRA_model.h5')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return 'No file uploaded!'
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected!'
        
        if file:
            # Process the audio file using your pipeline
            processed_audio = detect_black_rail_calls(file)
            
            # Predict using your model
            prediction = model.predict(processed_audio)
            
            # Here, you can format the prediction as you like
            # For simplicity, let's assume a binary classification
            result = "Black rail detected!" if prediction[0] > 0.95 else "No black rail detected."
            
            return result

    return '''
    <!doctype html>
    <title>Upload Audio File</title>
    <h1>Upload an audio file and check for black rail calls</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)

