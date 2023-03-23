
import cv2
import numpy as np
import math
import os
import re
import torch
import collections
from collections import Counter

def oned_entropy(picturepath):
        tmp = []
        for i in range(256):
                tmp.append(0)
        val = 0
        k = 0
        res = 0
        image = cv2.imread(picturepath,0)
        img = np.array(image)
        for i in range(len(img)):
                for j in range(len(img[i])):
                        val = img[i][j]
                        tmp[val] = float(tmp[val] + 1)
                        k =  float(k + 1)
        for i in range(len(tmp)):
                tmp[i] = float(tmp[i] / k)
        for i in range(len(tmp)):
                if(tmp[i] == 0):
                        res = res
                else:
                        res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
        print(res)
        return res

def two_entropy(picturepath):
        img = cv2.imread(picturepath, flags=cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img)
        compare_list = []
        for m in range(1,img.size()[0]-1):
                for n in range(1,img.size()[1]-1):
                        sum_element = img[m-1, n-1] + img[m-1, n] + img[m-1, n+1] + img[m, n-1] + img[m, n+1] + img[m+1, n-1] + img[m+1, n] + img[m+1, n+1]
                        sum_element = int(sum_element)
                        mean_element = sum_element//8
                        pix = int(img[m, n])
                        temp = (pix, mean_element)
                        compare_list.append(temp)
        
        # print(compare_list)
        compare_dict = collections.Counter(compare_list)
        H = 0.0
        for freq in compare_dict.values():
                f_n2 = freq / img.size()[0]**2
                log_f_n2 = math.log(f_n2)
                h = -(f_n2 * log_f_n2)
                H += h
 
        print(H)

dir = './attention_map2/test/'
pictures = os.listdir(dir)
pictures.sort()
entropy = []
seg = []
for pic in pictures:
        number = re.findall(r'\d+',pic)
        print(number)
for pic in pictures:
        number = re.findall(r'\d+',pic)
        seg.append(number)
        entropy.append(two_entropy(dir+pic))
print(np.array(seg).shape, np.array(entropy).shape)
# one_entropy = np.hstack((np.array(seg),np.array(entropy)))
print(seg,entropy)
