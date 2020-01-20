from django.shortcuts import render
from django.utils.encoding import smart_str
from django.http import HttpResponse

from zipfile import ZipFile
from PIL import Image, ImageFile
import numpy as np
import base64
import os
from io import BytesIO

from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing  import image 
from keras.preprocessing import image

all_classes = load_model('./neural_networks/nn_07.h5')
# person_rec.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def index(request):
    if request.method=='POST':
        try:
            zip_file = request.FILES['images']
            lookupTable = []
            lookupTable.append(bool(request.POST.get('ContainsCat',False)))
            lookupTable.append(bool(request.POST.get('ContainsDog',False)))
            lookupTable.append(bool(request.POST.get('ContainsPerson',False)))
            
            images_meeting_condition = []
            rest_of_images = []

            with ZipFile(zip_file,'r') as archive:
                for entry in archive.infolist():
                    with archive.open(entry) as file:
                        img = image.load_img(file)
                        img_resized = img.resize((120,120))
                        x = image.img_to_array(img_resized)
                        x = np.array([x,]) * (1.0/255.0)
                        results = []

                        for i in all_classes.predict([x,])[0]:
                            if i > 0.5:
                                results.append(True)
                            else:
                                results.append(False)

                        meets_requirements = (lookupTable == results)
                        if meets_requirements:
                            images_meeting_condition.append([file.name,img])
                        else:
                            rest_of_images.append([file.name,img])

            response = HttpResponse(content_type='application/zip')
            archive =  ZipFile(response,"w")
            for pair in images_meeting_condition:
                pair[1].save(pair[0],"JPEG")
                archive.write(pair[0],"positive\\"+pair[0])
                os.remove(pair[0])
            for pair in rest_of_images:
                pair[1].save(pair[0],"JPEG")
                archive.write(pair[0],"negative\\"+pair[0])
                os.remove(pair[0])

            
            response['Content=Disposition'] = f'attachment; filename="res.zip"'
            return response
        except Exception as e:
            return render(request, 'image_segregator/index.html',{'Error':e})
    else:
        return render(request, 'image_segregator/index.html')
    