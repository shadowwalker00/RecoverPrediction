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
import cv2

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
    return train_out,target_out

def feature_extraction(image):
    # functionality: input an image and output its feature extraction
    # note: be careful with using which layer of the feature
    graph = tf.Graph()
    with graph.as_default():
        images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
        network = Network(weight_path)
        c1,c2,c3,c4,conv53,reshape_vec_51,reshape_vec_52,reshape_vec_53 = network.inference(images_tf) 
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()        
        image = image[np.newaxis, :]
        reshape_vec_val = sess.run(reshape_vec_52,feed_dict={images_tf: np.array(image)})
    print("============== Finish extracting the feature: {}=================".format(reshape_vec_val.shape))
    return reshape_vec_val

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
    myDict = {"happy": [260,0.01], "unhappy": [260 ,0.001],"friendly":[260,0.0001], "unfriendly": [260,0.0001], "sociable": [260,0.01], "introverted": [220,0.001],
    "attractive": [260,0.00001], "unattractive": [260,0.001],"kind": [260, 0.00001], "mean": [180,0.0001], "caring": [220,0.00001], "cold": [260, 0.0001], 
    "trustworthy": [220, 0.01],  "untrustworthy": [180, 0.01],  "responsible": [180, 0.00001], "irresponsible": [260,0.0001], "confident": [260,0.001], 
    "uncertain":[220,0.001], "humble": [140,0.01], "egotistic": [180, 0.001], "emotStable": [220, 0.01], "emotUnstable": [260, 0.0001],
    "normal": [180, 0.00001], "weird":[220, 0.001], "intelligent":[180, 0.0001], "unintelligent":[180, 0.01],"interesting":[180, 0.001], "boring": [220, 0.01],
    "calm":[220, 0.01], "aggressive":[140, 0.001], "emotional": [100, 0.0001], "unemotional":[220, 0.00001], "memorable": [140, 0.01], "forgettable": [140, 0.01],
    "typical": [140, 0.00001], "atypical": [220, 0.00001], "common": [100, 0.01], "uncommon":[140, 0.0001], "familiar": [140, 0.001], "unfamiliar":[140, 0.001]
    }    
    for key, value in myDict.items():        
        train_target = target[:, label_list.index(key)]    
        trainX = train_out
        pcaObj = trainPCA(trainX, value[0])
        with open("./model/pca_"+key+"_conv52_mis.pkl","wb") as file:
            Pickle.dump(pcaObj,file)
        transformed_trainX = pcaObj.transform(trainX)        
        predictor = trainPredictor(transformed_trainX,train_target,value[1])

        with open("./model/pred_"+key+"_conv52_mis.pkl","wb") as file:
            Pickle.dump(predictor,file)
        print("{} saved pca to {} and pred to {}".format(key,"./model/pca_"+key+".pkl","./model/pred_"+key+".pkl"))


def predictFunction(images, attribute):
    #predict image wrt given attribute
    #images: a list of numpy array
    #attribute: name of the attribute 
    #return: a list of prediction value
    model_file = ["./model/bestModels/pca_"+attribute+".pkl","./model/bestModels/pred_"+attribute+".pkl"]
    pred_list = []
    for image in images:
        cropped_image = image
        out_feture = feature_extraction(cropped_image)
        with open(model_file[0],'rb') as f:
            pcaObj = pickle.load(f)
        with open(model_file[1],'rb') as f:
            predictObj  = pickle.load(f)
        image_data = pcaObj.transform(out_feture)
        pred_list.append(predictObj.predict(image_data))
    return pred_list



def testTestset():
    # funtionality: apply the models to the MIT2k testset and find out the performance
    # output: an excel file storing the result
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
        print(pho)
        accs.append(pho)
        print("testing with {}".format(label))
    #save_excel(label_list,accs,"conv53_test.xlsx")


def testCelebA(attribute):
    # functionality: test bestModels on CelebA given a specific attribute
    celebA_file = "/home/ghao/Prediction_Face/testCelebA"
    picture_dir_path = "/raid/amanda/celebA/oneface"
    with open(celebA_file+"/"+attribute+"/CelebA_"+attribute+".pickle", 'rb') as handle:
        data = pickle.load(handle)

    all_result = []
    for index,item in enumerate(data):
        image_path, gt_value = item
        print(os.path.join(picture_dir_path,image_path))        
        loadout_image = load_image(os.path.join(picture_dir_path,image_path))
        print(np.max(loadout_image))
        print(np.min(loadout_image))
        skimage.io.imsave("name.png", loadout_image)
        image_list = [loadout_image]
        pred_result = predictFunction(image_list,attribute)
        all_result.extend(pred_result)
    return all_result


def testMIT2k(attribute):
    # functionality: test bestModels on CelebA given a specific attribute
    MIT2k_file = "/home/ghao/Prediction_Face/datasets"
    picture_dir_path = "/home/ghao/Prediction_Face/datasets/images"
    with open(MIT2k_file+"/MIT2k_"+attribute+".pickle", 'rb') as handle:
        data = pickle.load(handle)

    all_result = []
    gt_list = []
    for index,item in enumerate(data):
        image_path, gt_value = item
        print(os.path.join(picture_dir_path,image_path))        
        loadout_image = load_image(os.path.join(picture_dir_path,image_path))                
        image_list = [loadout_image]
        pred_result = predictFunction(image_list,attribute)
        all_result.extend(pred_result)
        gt_list.append(gt_value)
    return all_result, gt_list




def testTestsetDebug():
    # funtionality: apply the models to the MIT2k testset and find out the performance
    # output: an excel file storing the result
    testset = pd.read_pickle(testset_path)
    testout, testlabel = forward(testset)
    label = pd.read_pickle(label_path)
    label_list = label["labelname"].values.tolist()
    accs = []
    for index,label in enumerate(label_list):
        print(label)
        if label == "aggressive":        	
            with open("./model/pca_"+label+"_conv53.pkl","rb") as file:
                pcaObj = Pickle.load(file)
            with open("./model/pred_"+label+"_conv53.pkl","rb") as file:
                predictor = Pickle.load(file)        
            transformed_testX = pcaObj.transform(testout)   
            predict_test = predictor.predict(transformed_testX)
            pho, _ = stats.stats.spearmanr(predict_test, testlabel[:, index])
            print(pho)
            accs.append(pho)
            print("testing with {}".format(label))
    #save_excel(label_list,accs,"conv53_test.xlsx")


def trainAgainDebug(train_out,target,label_list):
    myDict = {"happy": [260,0.01], "unhappy": [260 ,0.001],"friendly":[260,0.0001], "unfriendly": [260,0.0001], "sociable": [260,0.01], "introverted": [220,0.001],
    "attractive": [260,0.00001], "unattractive": [260,0.001],"kind": [260, 0.00001], "mean": [180,0.0001], "caring": [220,0.00001], "cold": [260, 0.0001], 
    "trustworthy": [220, 0.01],  "untrustworthy": [180, 0.01],  "responsible": [180, 0.00001], "irresponsible": [260,0.0001], "confident": [260,0.001], 
    "uncertain":[220,0.001], "humble": [140,0.01], "egotistic": [180, 0.001], "emotStable": [220, 0.01], "emotUnstable": [260, 0.0001],
    "normal": [180, 0.00001], "weird":[220, 0.001], "intelligent":[180, 0.0001], "unintelligent":[180, 0.01],"interesting":[180, 0.001], "boring": [220, 0.01],
    "calm":[220, 0.01], "aggressive":[140, 0.001], "emotional": [100, 0.0001], "unemotional":[220, 0.00001], "memorable": [140, 0.01], "forgettable": [140, 0.01],
    "typical": [140, 0.00001], "atypical": [220, 0.00001], "common": [100, 0.01], "uncommon":[140, 0.0001], "familiar": [140, 0.001], "unfamiliar":[140, 0.001]
    }    
    for key, value in myDict.items():        
        if key == "aggressive":
            train_target = target[:, label_list.index(key)]    
            trainX = train_out
            pcaObj = trainPCA(trainX, value[0])
            with open("./model/pca_"+key+"_conv52_mis.pkl","wb") as file:
                Pickle.dump(pcaObj,file)
            transformed_trainX = pcaObj.transform(trainX)        
            predictor = trainPredictor(transformed_trainX,train_target,value[1])

            with open("./model/pred_"+key+"_conv52_mis.pkl","wb") as file:
                Pickle.dump(predictor,file)
            print("{} saved pca to {} and pred to {}".format(key,"./model/pca_"+key+".pkl","./model/pred_"+key+".pkl"))



if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    # label = pd.read_pickle(label_path)
    # label_list = label["labelname"].values.tolist()
    # #get data

    # trainset = pd.read_pickle(trainset_path)
    # train_out, target = forward(trainset)
    # print("Obtaining training vector from conv5:shape={}".format(train_out.shape))

    # #select parameters    
    # for i in range(len(label_list)):
    #     ks, alphas, accs = trainProcess(i,train_out,target)
    #     save_result(ks, alphas, accs, label_list[i])

    # # #train again best model
    # # trainAgain(train_out,target,label_list)

    # # #test model
    # # testTestset()

    # result = testCelebA("aggressive")
    # with open("test.pickle", 'wb') as handle:
    #     pickle.dump(result, handle)


    


    # label = pd.read_pickle(label_path)
    # label_list = label["labelname"].values.tolist()
    # testset = pd.read_pickle(testset_path)
    # image_name_list = testset["image_path"].values
    # train_out, target = forward(testset)
    # with open("testMIT2k.pickle", 'wb') as handle:
    #     result = zip(image_name_list,target[:,label_list.index("aggressive")])
    #     pickle.dump(result, handle)


    mit_result, mit_gt = testMIT2k("aggressive")
    with open("testMIT2k_aggressive_scatter.pickle", 'rb') as handle:
        data = pickle.load(handle)
    print("Finish")
    
