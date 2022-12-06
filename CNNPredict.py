# from fastai.vision.all import *
from fastai.vision import *
from PIL import Image
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders

# proj_path = './Training/'
# p_path = Path(proj_path)

# # Splitting images into 80/20 and transforming/normalizing them
# # np.random.seed(9)
# data = ImageDataLoaders.from_folder(p_path, valid_pct=0.2)
# learn = cnn_learner(data, models.resnet34, metrics=error_rate).load("model1_small")
# # learn = cnn_learner(data, models.resnet34)
calories_dict = {'Apple Braeburn': '71 calories', 
'Carambula': '31 calories', 
'Eggplant': '25 calories', 
'Lemon': '29 calories', 
'Pear Abate': '40 calories', 
'Apple Red Delicious': '80 calories', 
'Cherry 1': '50 calories', 
'Fig': '37 calories', 
'Mandarine': '47 calories', 
'Pear Stone': '71 calories',
'Avocado': '107 calories',
'Chestnut': '131 calories',
'Guava': '68 calories',
'Mango': '60 calories',
'Banana': '89 calories',
'Clementine': '47 calories',
'Kaki': '127 calories',
'Peach': '68 calories'}

name_dict = {'Apple Braeburn': 'Apple', 
'Carambula': 'Carambula', 
'Eggplant': 'Eggplant', 
'Lemon': 'Lemon', 
'Pear Abate': 'Pear', 
'Apple Red Delicious': 'Apple', 
'Cherry 1': 'Cherry', 
'Fig': 'Fig', 
'Mandarine': 'Mandarine', 
'Pear Stone': 'Pear',
'Avocado': 'Avocado',
'Chestnut': 'Apple',
'Guava': 'Guava',
'Mango': 'Mango',
'Banana': 'Banana',
'Clementine': 'Clementine',
'Kaki': 'Kaki',
'Peach': 'Peach'}

categories = ('Apple Braeburn', 'Carambula', 'Eggplant', 'Lemon', 'Pear Abate', 'Apple Red Delicious', 'Cherry 1', 'Fig', 'Mandarine', 'Pear Stone', 'Avocado', 'Chestnut', 'Guava', 'Mango', 'Banana', 'Clementine', 'Kaki', 'Peach')

def predict():
    path = "./model-small.pkl"
    learn = load_learner(path)

    upload_file = "./1-image.jpg"
    im = PILImage.create(upload_file)
    im.thumbnail ((100, 100))

    pred,idx,probs = learn.predict(im)
    # return print(dict(zip(categories, map(float, probs))))
    return {"name": pred, "calories": calories_dict[pred]}
    # return print(pred, calories_dict[pred])

# print(predict())