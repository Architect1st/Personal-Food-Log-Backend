from flask import Flask, jsonify, request
# import base64
import os
from CNNPredict import predict
from PIL import Image, ImageEnhance

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def submit():
    if request.method == "POST":
        upload_file = request.files['file']
        # upload_file = crop_max_square(upload_file).resize((2186, 2186), Image.LANCZOS)
        file_name = upload_file.filename

        # crop the image to a square
        upload_file = Image.open(upload_file)
        
        # enhancer = ImageEnhance.Contrast(upload_file)
        # factor = 0.5 #increase contrast
        # upload_file = enhancer.enhance(factor)

        # enhancer = ImageEnhance.Brightness(upload_file)
        # factor = 1.5 #increase brightness
        # upload_file = enhancer.enhance(factor)

        box = (1019, 0, 3205, 2186)
        upload_file = upload_file.crop(box)
        # upload_file.show()

        # file_path = r"./4/"
        file_path = r"./"
        if upload_file:
            file_paths = os.path.join(file_path, file_name)
            upload_file.save(file_paths)
            # return {"name":"Obviously apple", "calories":"234"}
            return predict()

app.run(debug=True, host="0.0.0.0", port = 80)