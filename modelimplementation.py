from flask import Flask, render_template, request
import os


model = load_model('D:\\Untitled Folder\\model_byprince.h5')


app  = Flask(__name__, template_folder='templates')

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')
@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['filename']
    imgname = imagefile.filename
    dif = imgname[0]
    if(dif == 'c'):
         image_path = "./testit/cat/" + imagefile.filename
         imagefile.save(image_path)
    else:
         image_path = "./testit/dog/" + imagefile.filename
         imagefile.save(image_path)
        
    img_testit = 'D:\\Untitled Folder\\testit'
    test_image = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=img_testit, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)
    y_pred = model.predict(x=test_image, steps=len(test_image), verbose=0)
    y_pred  = np.round(y_pred)
    imgs, labels = next(test_image)
    res = ""
    for item in y_pred:
        if item[0] == 1:
            res = "This is a Cat"
            os.remove(image_path)
            
        else:
            res = "This is a Dog"
            os.remove(image_path)
    
   
    
    return render_template('index.html',prediction=res)
   

if __name__ == '__main__':
    app.run(port=3000,debug=True, use_reloader=False)