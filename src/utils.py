import numpy as np
import pickle
import os
import pandas as pd
import random
import argparse
import skimage.io
import skimage.transform
import xlsxwriter

	
def splitTrainVal(trainset,fold):
	index = list(range(4000))
	val_index = index[fold*800: fold*800+800]
	train_index = index[:fold*800]+index[fold*800+800:]
	train = trainset.loc[train_index]
	validation = trainset.loc[val_index]
	return train,validation

def load_image( path ):
    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    img /= 255.
    resized_img = skimage.transform.resize(img,[224,224],mode='constant')
    return resized_img

def save_result(ks, alphas, accs, filename):
    workbook  = xlsxwriter.Workbook("./"+filename+".xlsx")
    worksheet = workbook.add_worksheet(filename)
    headings = ['K','Alpha','Accuracy']
    worksheet.write_row('A1',headings)
    worksheet.write_column('A2',ks)
    worksheet.write_column('B2',alphas)
    worksheet.write_column('C2',accs) 

def save_excel(labels, accs):
    filename = "./test.xlsx"
    workbook  = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet("testResult")
    headings = ['Label','Accuracy']
    worksheet.write_row('A1',headings)
    worksheet.write_column('A2',labels)
    worksheet.write_column('B2',accs)

if __name__=="__main__":
    #create dataset
    parser = argparse.ArgumentParser()
    parser.description='Dataset parser'
    parser.add_argument('--name', help="name of dataset")
    allPara = parser.parse_args()
    data = Dataset()
    data.createDataSet(allPara.name)