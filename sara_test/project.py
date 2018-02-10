import os.path
import os
import cv2

import matplotlib.pyplot as plt

from skimage import img_as_ubyte
from skimage.measure import label
from skimage.morphology import skeletonize
from skimage.measure import regionprops

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata


def pozicijaTacke(x, y):
    yy = y2 + ((x - x2) * (y1-y2)) / (x1 - x2)
    return yy - y

def prolazIzaPrave(bbox):
    if (bbox[1] > x2 or bbox[3] + 4 < x1):
        return False
    if (bbox[0] > y1 or  bbox[2] + 4 < y2):
        return False

    if (trenutniFrame > 0 ):
        if (pozicijaTacke(bbox[3] + 1, bbox[2] + 1) <= 0):
            return False
    else:
        if (pozicijaTacke(bbox[1], bbox[0]) < 0):
            return False
    if (pozicijaTacke( bbox[3] + 4, bbox[2] + 4) > 0):
        return False

    return True

def obukaDataseta(dataSet):
    opseg_br = range(0, len(dataSet));
    for i in opseg_br:
        num=dataSet[i].reshape(28, 28);
        sliq = cv2.inRange(num, 150, 255)
        output = cv2.bitwise_and(num, num)

         #show the images
        #cv2.imshow("images", np.hstack([dataSet[i].reshape(28, 28), output]))
        cv2.waitKey(0)

        clos = cv2.morphologyEx(sliq, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        regioni = regionprops(label(clos))
        if (len(regioni) <= 1):
            bbox = regioni[0].bbox
        else:
            max_visina = 0
            for region in regioni:
                t_bbox = region.bbox
                t_visina = region.bbox[2] - region.bbox[0]
                if (t_visina > max_visina):
                      bbox = t_bbox
                      max_visina = t_visina

        # napravi sliku
        vrste = range(bbox[0], bbox[2])
        kolone = range(bbox[1], bbox[3])
        sliqq = np.zeros((28, 28))

        x = 0
        for vrsta in vrste:
            y = 0
            for kolona in kolone:
                sliqq[x, y] = num[vrsta, kolona]
                y = y + 1
            x = x + 1
        dataSet[i] = sliqq.reshape(1, 784)

def kreiranjeListeBr(bbox, brojevi_list, br):
    prvi_ulaz = True;
    for trenutni_br in brojevi_list:
        if (trenutni_br[0] == br and bbox[0] +7 > trenutni_br[2]  and bbox[1] + 7 > trenutni_br[1]):
             brojevi_list.remove(trenutni_br)
             prvi_ulaz = False;
             brojevi_list.append((br, bbox[1], bbox[0]))
    if (prvi_ulaz == True):
         print 'U frejmu ' + str(trenutniFrame) + '. pronasao br: ' + str(br)
         brojevi_list.append((br, bbox[1], bbox[0]))

def nasaoBroj(bbox, slika):
    visina = reg.bbox[2] - reg.bbox[0]
    sirina = reg.bbox[3] - reg.bbox[1]
    poX = range(0, visina)
    poY = range(0, sirina)
    slika_br = np.zeros((28, 28))
    for x in poX:
        for y in poY:
            slika_br[x, y] = slika[bbox[0] + x - 1, bbox[1] + y - 1 ]
    return slika_br


mnist = fetch_mldata('MNIST original')
putanja = 'C:\\Users\\Admin\\Desktop\\sara_test'
mnistFile = 'dataSet'
mnistPutanja = 'C:\\Users\\Admin\\Desktop\\sara_test\\DataSet'
file = os.path.join(mnistPutanja, mnistFile)

if (os.path.exists(file) == True):
    ucitanFile = np.load(file)
else:
    ucitanFile = mnist.data
    obukaDataseta(ucitanFile)
    np.save(os.path.join(mnistPutanja, mnistFile), ucitanFile)

kneCLas = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
ktrain = kneCLas.fit(ucitanFile, mnist.target);

putanja = putanja + '\\Videi'

for imeV in os.listdir(putanja):
    if (os.path.isfile(os.path.join(putanja, imeV))):
          video_ime = os.path.join(putanja, imeV)
    trenutniFrame = 0
    brojevi_list = []
    cap = cv2.VideoCapture(video_ime)
    while (cap.isOpened()):
        ret, frame = cap.read()

        if (ret == False):
            break
        siva = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', siva)
        if (trenutniFrame == 0 ):
            edges = cv2.Canny(siva, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            x1 = lines[0][0][0]
            y1 = lines[0][0][1]
            x2 = lines[0][0][2]
            y2 = lines[0][0][3]
        img = cv2.inRange(siva, 163, 255)
        clos = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        labela = label(clos)
        regioni = regionprops(labela)
        for reg in regioni:
            if ((reg.bbox[2] - reg.bbox[0]) <= 10 or prolazIzaPrave(reg.bbox) == False):
                continue
            img_br = nasaoBroj(reg.bbox, siva)
            prepoznat_br = int(ktrain.predict(img_br.reshape(1, 784)))
            kreiranjeListeBr(reg.bbox, brojevi_list, prepoznat_br)
        trenutniFrame += 1

    suma = 0
    for broj in brojevi_list:
        suma += broj[0]

    print 'Za video ' + (imeV) + ' suma brojeva iznosi: ' + str(suma) + '\n'

cap.release()
cv2.destroyAllWindows()