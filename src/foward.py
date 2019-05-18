from datetime import datetime
import pytz
import time
import os
import pandas as pd
import numpy as np
import pickle as Pickle
import tensorflow as tf
import scipy.stats as stats
from network import Network
import argparse
from utils import *
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from facedetector import FaceDetector

root_path = '/home/ghao/Prediction_Face'
image_path = os.path.join(root_path,'datasets/images')
trainset_path = os.path.join(root_path,'datasets/train.pickle')
testset_path = os.path.join(root_path,'datasets/test.pickle')
label_path = os.path.join(root_path,'datasets/label.pickle')
model_path = os.path.join(root_path,'trained_models/VGG/face')
out_path = os.path.join(root_path,'out/')
weight_path = os.path.join(root_path,'pretrained_weight/VGG/caffe_layers_value.pickle')


def forward(trainset):    
    #trainset = pd.read_pickle(trainset_path)
    batch_size = 40    
    graph = tf.Graph()
    with graph.as_default():
        images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
        network = Network(weight_path)
        c1,c2,c3,c4,conv53,reshape_vec_51,reshape_vec_52,reshape_vec_53 = network.inference(images_tf) 
        saver = tf.train.Saver()

    train_out = None
    target_out = None
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for start, end in zip(range(0, len(trainset)+batch_size, batch_size),range(batch_size, len(trainset)+batch_size, batch_size)):
            current_data = trainset[start:end]
            current_image_paths = current_data['image_path'].values                     
            current_images = np.array(list(map(lambda x: load_image(os.path.join(image_path,x)), current_image_paths))) #load image
            good_index = np.array(list(map(lambda x: x is not None, current_images)))   
            current_data = current_data[good_index]          
            current_image_paths = current_image_paths[good_index]    
            current_images = np.stack(current_images[good_index])

            current_labels = np.array(current_data['label'].values)
            
            current_labels_deal = np.zeros((current_labels.shape[0],40))
            for index,row in enumerate(current_labels):
                current_labels_deal[index,:] = row
            if target_out is None:
                target_out = current_labels_deal
            else:
                target_out = np.vstack((target_out,current_labels_deal))

            reshape_vec_val = sess.run(reshape_vec_53,feed_dict={images_tf: current_images})

            if train_out is None:
                train_out = reshape_vec_val
            else:
                train_out = np.vstack((train_out,reshape_vec_val))
        print(train_out.shape,target_out.shape)
    return train_out,target_out


def trainPCA(train,k):
    pca = PCA(n_components=k)
    pca.fit(train)
    return pca

def trainPredictor(trainX,target,alpha):
    clf = Ridge(alpha=alpha)
    clf.fit(trainX, target)
    return clf

def trainProcess(index_attribute,train_out,target):
    target = target[:, index_attribute]
    index = list(range(4000))
    ks = []
    alphas = []
    accs = []
    for select_k in range(100,300,40):
        for select_alpha  in [0.00001,0.0001,0.001,0.01]:
            print("Fine tune k={}, alpha = {}".format(select_k,select_alpha))
            average_performance = 0
            for fold in range(5):
                val_index = index[fold*800: fold*800+800]
                train_index = index[:fold*800]+index[fold*800+800:]
                #split data
                trainX = train_out[train_index]
                train_target = target[train_index]

                valX = train_out[val_index]
                val_target = target[val_index]
                pcaObj = trainPCA(trainX,select_k)

                transformed_trainX = pcaObj.transform(trainX)
                predictor = trainPredictor(transformed_trainX,train_target,select_alpha)
                transformed_valX = pcaObj.transform(valX)
                predict_val = predictor.predict(transformed_valX)
                pho, _ = stats.stats.spearmanr(predict_val, val_target)
                print(pho)
                average_performance += pho
            average_performance /= 5
            print("The average_performance is {}".format(average_performance))
            ks.append(select_k)
            alphas.append(select_alpha)
            accs.append(average_performance)
    return ks, alphas, accs

def trainAgain(train_out,target,label_list):
    myDict = {"happy": [190,0.01], "unhappy": [190,0.01],"friendly":[280, 0.00001], "unfriendly": [280,0.01], "sociable": [100, 0.00001], "introverted": [130, 0.0001],
    "attractive": [280, 0.00001], "unattractive": [190,0.00001],"kind": [280,0.00001], "mean": [130, 0.0001], "caring": [280,0.001], "cold": [190,0.00001], 
    "trustworthy": [280, 0.001],  "untrustworthy": [100, 0.01],  "responsible": [160, 0.001], "irresponsible": [220, 0.00001], "confident": [100, 0.001], 
    "uncertain":[100, 0.0001], "humble": [280, 0.00001], "egotistic": [280, 0.0001], "emotStable": [130, 0.0001], "emotUnstable": [130, 0.0001],
    "normal": [220, 0.01], "weird":[190, 0.001], "intelligent":[160, 0.00001], "unintelligent":[130, 0.001],"interesting":[280, 0.00001], "boring": [130, 0.0001],
    "calm":[220, 0.01], "aggressive":[280, 0.0001], "emotional": [100, 0.0001], "unemotional":[160, 0.00001], "memorable": [100, 0.01], "forgettable": [100, 0.01],
    "typical": [280, 0.00001], "atypical": [100, 0.00001], "common": [100, 0.01], "uncommon":[100, 0.001], "familiar": [100, 0.01], "unfamiliar":[100,0.00001]
    }    
    for key, value in myDict.items():        
        train_target = target[:, label_list.index(key)]    
        trainX = train_out
        pcaObj = trainPCA(trainX, value[0])
        with open("./model/pca_"+key+".pkl","wb") as file:
            Pickle.dump(pcaObj,file)
        transformed_trainX = pcaObj.transform(trainX)        
        predictor = trainPredictor(transformed_trainX,train_target,value[1])

        with open("./model/pred_"+key+".pkl","wb") as file:
            Pickle.dump(predictor,file)
        print("{} saved pca to {} and pred to {}".format(key,"./model/pca_"+key+".pkl","./model/pred_"+key+".pkl"))


def predict(images, attribute):
    #predict image wrt given attribute
    #images: a list of numpy array
    #attribute: name of the attribute 
    #return: a list of prediction value
    model_file = ["./model/pca_"+attribute+".pkl","./model/pred_"+attribute+".pkl"]
    pred_list = []
    for image in images:
        facedetect = FaceDetector()
        cropped_image = facedetect.detection(image)

        with open(model_file[0],'rb') as f:
            pcaObj = pickle.load(f)
        with open(model_file[1],'rb') as f:
            predictObj  = pickle.load(f)

        image_data = pcaObj.transform(cropped_image)
        pred_list.append(predictObj.predict(image_data))
    return pred_list



def testTestset():
    testset = pd.read_pickle(testset_path)
    testout, testlabel = forward(testset)
    label = pd.read_pickle(label_path)
    label_list = label["labelname"].values.tolist()

    accs = []
    for index,label in enumerate(label_list):
        with open("./model/pca_"+label+".pkl","rb") as file:
            pcaObj = Pickle.load(file)
        with open("./model/pred_"+label+".pkl","rb") as file:
            predictor = Pickle.load(file)        
        transformed_testX = pcaObj.transform(testout)   
        predict_test = predictor.predict(transformed_testX)
        pho, _ = stats.stats.spearmanr(predict_test, testlabel[:, index])
        accs.append(pho)
        print("testing with {}".format(label))
    save_excel(label_list,accs)




if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    label = pd.read_pickle(label_path)
    label_list = label["labelname"].values.tolist()
    #get data

    trainset = pd.read_pickle(trainset_path)
    train_out, target = forward(trainset)
    print("Obtaining training vector from conv5:shape={}".format(train_out.shape))

    for i in range(len(label_list)):
        ks, alphas, accs = trainProcess(i,train_out,target)
        save_result(ks, alphas, accs, label_list[i])

    #trainAgain(train_out,target,label_list)
    
