import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
def readfile(file):
    sd=[]
    with open(file, 'r') as f:
        for x in f:
            sd.append([float(n) for n in x.strip().split(',')])
    # sd = np.reshape(sd, -1)
    return sd
def readTestfile(file):
    sd=[]
    with open(file, 'r') as f:
        for x in f:
            sd.append([float(n) for n in x.strip().split(' ')])
    # sd = np.reshape(sd, -1)
    return sd

# file = "Image/sitT.png"
# img = cv2.imread("Image/sitT.png", 0)
def loaddata():
    img_names = sorted(glob.glob('Image/*.png'))
    Testing_img_names = sorted(glob.glob('Testdatasets/*.png'))
    # print(img_names)

    imges_data = []
    for x in range(len(img_names)):
        file = img_names[x]
        img = cv2.imread(file)
        # print(img.shape)
        img = cv2.resize(img,(42,42))
        imges_data.append(img)

    Test_imges_data = []
    for x in range(len(Testing_img_names)):
        fileT = Testing_img_names[x]
        imgT = cv2.imread(fileT)
        # print(img.shape)
        imgT = cv2.resize(imgT,(42,42))
        Test_imges_data.append(imgT)
    #
    imges_data = np.array(imges_data)
    Test_imges_data = np.array(Test_imges_data)
    # print(imges_data.shape)
    filelabel = "Outputdata/label.csv"
    labellist = readfile(filelabel)
    labellist = sorted(labellist,reverse=True)
    Testlabelfile = "Outputdata/Testlabel1.csv"
    Testlabellist = readTestfile(Testlabelfile)

    train_set_y = []
    for x in range(len(labellist)):
        train_set_y.append(labellist[x][0])
    test_set_y = []
    for x in range(len(Testlabellist)):
        test_set_y.append(Testlabellist[x][0])
    # print(test_set_y)

    train_set_y = np.array(train_set_y).reshape(1,-1)
    print(train_set_y)

    test_set_y = np.array(test_set_y).reshape(1,-1)
    print(test_set_y)

    train_set_x = imges_data
        # imges_data.reshape(imges_data.shape[0],-1).T
    # print('train_set_x_flatten shape:', train_set_x_flatten.shape)
    train_set_x_flatten = train_set_x/255

    test_set_x = Test_imges_data
        # Test_imges_data.reshape(Test_imges_data.shape[0],-1).T
    # print('test_set_x_flatten shape:', test_set_x_flatten.shape)
    test_set_x_flatten = test_set_x/255
    return train_set_x, train_set_y, test_set_x, test_set_y