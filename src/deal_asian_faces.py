import cv2
import numpy as np
import matplotlib.pyplot as plt
from facedetector import FaceDetector
import os
import xlsxwriter

def save_height_width(filenames, widths, heights):
    filename = "./height_width_asianfaces.xlsx"
    workbook  = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet("AsianFaces")
    headings = ['File','Width','Height']
    worksheet.write_row('A1',headings)
    worksheet.write_column('A2',filenames)
    worksheet.write_column('B2',widths)
    worksheet.write_column('C2',heights)
    print("Saved to File={}".format(filename))


def dealAsian():
    detector = FaceDetector()
    data_path  = "./datasets/asianFacesCategory"
    data_out_path = "./datasets/asianFaces"
    identities = os.listdir(data_path)
    new_filenames = []
    widths = []
    heights = []

    for person in identities:
        dirpath = data_path+"/"+person
        if person != ".DS_Store" and person!="._.DS_Store":
            files = os.listdir(dirpath)
            for file in files:
                print("=============================")
                image = cv2.imread(dirpath+"/"+file)
                if image is not None:
                    print("Load Dir={} and Image={}".format(person,file))
                    res = detector.detection(image)
                    if res is not None:
                        crop_image,w,h = res              
                        cv2.imwrite(data_out_path+"/"+person+"_"+file,crop_image)
                        new_filenames.append(person+"_"+file)
                        widths.append(w)
                        heights.append(h)
                    else:
                        print("No Faces")                    

                else:
                    print("Read Image Wrong....")
    save_height_width(new_filenames,widths,heights)
if __name__ == "__main__":
    detector = FaceDetector()
    #image = cv2.imread("./test3.jpeg")
    dealAsian()
    print("Finish")

