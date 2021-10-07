import sys
import cv2
import face_recognition
import pickle
import os

name_array = {'Ijmert': 11, 'Bert': 5, 'Andreas': 0, 'Paco':0, 'Steven':0, 'Mathias':0}


def countDigits(string):
    digit_count = 0
    digit_array = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for c in string:
        if c in digit_array:
            digit_count += 1

    return digit_count

try:
    f = open("ref_embed.pkl", "rb")
    embed_dict = pickle.load(f)
    f.close()
except:
    embed_dict={}


name_dict = {}
dirs = os.listdir('images/')
for filename in dirs:
    name = filename[:filename.index('.jpg') - countDigits(filename)]
    if name not in name_dict:
        name_dict[name] = 1
    else:
        name_dict[name] += 1


for name in name_dict:
    if name not in embed_dict:
        try:
            for i in range(1, name_dict[name]+1):
                image_string = "images/%s%i.jpg" % (name, i)
                print(image_string)
                image = cv2.imread(image_string)
                face_locations = face_recognition.face_locations(image)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image)[0]
                if name in embed_dict:
                    embed_dict[name] += [face_encoding]
                else:
                    embed_dict[name] = [face_encoding]

        except:
            print("Something went wrong embedding %s" %name)

if embed_dict != {}:
    f = open("ref_embed.pkl", "wb")
    pickle.dump(embed_dict, f)
    f.close()
