from glob import glob
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image, ImageChops
import arabic_reshaper
from bidi.algorithm import get_display
import pandas as pd

sz=(301,295)
path='data/ar_val/'

emptyId = cv.imread('emptyId.jpg')
emptyId=cv.resize(emptyId,sz)

sh_data='effects/'
sh_data=glob(sh_data+'*')
#cv.imshow("id", emptyId[75:105, 150:278, :])

yDim = ['fName', 'lName', 'fAddress', 'lAddress', 'idNumber']
yDict ={'fName':75, 'lName':105, 'sAddress': 135, 'rAddress': 165, 'idNumber': 225 }

fname_pos=(100,80)
lname_pos=(110,115)
fadd_pos=(220,155)
ladd_pos=(110,180)
nn_pos=(100,260)
lengths=[(210,80),(110,115),(220,155),(110,180),(100,260)]

SHOW=True

def show(t,img):
    if not SHOW:return
    cv.imshow(t,img)

def skew(img):
    rows, cols, ch = img.shape
    pts1 = np.float32(
        [[cols*.25, rows*.95],
         [cols*.90, rows*.95],
         [cols*.10, 0],
         [cols,     0]]
    )
    pts2 = np.float32(
        [[cols*0.3, rows],
         [cols,     rows],
         [0,        2],
         [cols,     0]]
    )
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img, M, (cols, rows))
    return dst

def get_salt(shape):
    row,col,ch = shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.zeros(shape,np.uint8)

    # just adding Salt mode as out is zeros (pepper)
    num_salt = np.ceil(amount * np.product(shape) * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in shape]
    out[coords] = random.uniform(100,150)

    return out


def get_shadow(shape):
    shadow=cv.imread(random.choice(sh_data))
    shadow=cv.resize(shadow,(shape[1],shape[0]))

    # flipping
    if random.uniform(0,1)<0.5:
        shadow=cv.flip(shadow,0)
    if random.uniform(0,1)<0.5:
        shadow=cv.flip(shadow,1)

    return shadow

def blur(img):
    if random.uniform(0,1)>=0.4:return img
    f=random.choice([3,5,7,7])
    return cv.GaussianBlur(img,(f,f),0)

def add_img(img,ref):
    beta=random.uniform(0,.5)
    return cv.addWeighted(img,1.,ref,beta,0)

def play_img(img):
    if random.uniform(0,1)<0.8:return img
    shadow=get_shadow(img.shape)
    img=add_img(img,shadow)

    # add pepper with &
    salt=get_salt(img.shape)
    img=add_img(img,salt)

    img=blur(img)
    return img


names = pd.read_csv("ar_dict.txt")
name_numpy = names.to_numpy()

fontpath = "arialBlack.ttf"
font = ImageFont.truetype(fontpath, 18)
b, g, r, a = 0, 0, 0, 0
count = 0
fileNameList = []
wordList = []

#image_numpy = name_numpy[:, 0]
name_numpy = name_numpy[:, 0]
# print(name_numpy)

idNumbers = '٠١٢٣٤٥٦٧٨٩'
reshaped_text = arabic_reshaper.reshape(name_numpy[count])
reshaped_text = get_display(reshaped_text, base_dir='R')

fileNameList=[]
wordList=[]
j=0

for i in range(300):
    labelString = ''
    texts=[]

    img_pil=Image.fromarray(emptyId)
    draw = ImageDraw.Draw(img_pil)
    for nametype in yDim:
        if nametype ==  'fName':
            labelString = random.choice(name_numpy)
            reshaped_text = arabic_reshaper.reshape(labelString)
            reshaped_text = get_display(reshaped_text, base_dir='R')
            totalWidth=font.getsize(reshaped_text)[0]
            draw.text((280-totalWidth,80),  reshaped_text,font=font, fill=(b, g, r, a))
            texts.append(labelString)

        elif nametype ==  'lName':
            labelString = (random.choice(name_numpy)+ ' ' + random.choice(name_numpy) + ' ' + random.choice(name_numpy))
            reshaped_text = arabic_reshaper.reshape(labelString)
            reshaped_text = get_display(reshaped_text, base_dir='L')
            totalWidth=font.getsize(reshaped_text)[0]
            draw.text((280-totalWidth,115),  reshaped_text,font=font, fill=(b, g, r, a))
            texts.append(labelString)

        elif nametype == 'idNumber':
            labelString = ''.join(random.choice(idNumbers) for i in range(14))
            reshaped_text = arabic_reshaper.reshape(labelString )
            reshaped_text = get_display(reshaped_text, base_dir='L')
            totalWidth=font.getsize(reshaped_text)[0]
            draw.text((280-totalWidth,260),  reshaped_text,font=font, fill=(b, g, r, a))
            texts.append(labelString)

        elif nametype=='fAddress':
            labelString = random.choice(name_numpy)+ ' ' + random.choice(name_numpy)
            reshaped_text = arabic_reshaper.reshape(labelString)
            reshaped_text = get_display(reshaped_text, base_dir='L')
            totalWidth=font.getsize(reshaped_text)[0]
            draw.text((280-totalWidth,155),  reshaped_text,font=font, fill=(b, g, r, a))
            texts.append(labelString)

        elif nametype=='lAddress':
            labelString = random.choice(name_numpy)+ ' ' + random.choice(name_numpy) + ' ' + random.choice(name_numpy) \
                + ' ' + random.choice(name_numpy)
            reshaped_text = arabic_reshaper.reshape(labelString)
            reshaped_text = get_display(reshaped_text, base_dir='L')
            totalWidth=font.getsize(reshaped_text)[0]
            draw.text((280-totalWidth,180),  reshaped_text,font=font, fill=(b, g, r, a))
            texts.append(labelString)


    img_pil = np.array(img_pil)
    #img=skew(img_pil)
    img_pil=play_img(img_pil)
    '''
    show('',img_pil)
    if cv.waitKey(0)==27:
        break
    '''

    margin=5
    for i in range(len(texts)):
        p2=lengths[i][1]
        totalWidth, totalHeight =font.getsize(texts[i])
        totalWidth = totalWidth + 3
        p1=280-totalWidth

        if 280-totalWidth<0:
            continue

        img=img_pil[p2-3:p2 + totalHeight+6, 280-totalWidth:283]#, :
        print(j,texts[i])
        #[p1-margin//2:p1+totalWidth[1]+margin//2,p2-margin:p2+totalWidth[0]+margin]

        #img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        cv.imwrite(path+str(j)+'.png',img)
        fileNameList.append(str(j)+'.png')
        wordList.append(texts[i])
        j+=1


dict = {'filename': fileNameList, 'words': wordList}#name_numpy.tolist()}
df = pd.DataFrame(dict)
df.to_csv(path+'labels.csv', index=False)
