import cv2
import torch
import os
import collections
from collections import Counter
import math
 
image_dir = './attention_map2/train/IMG_0846Seg_556_Health.jpg'
img = cv2.imread(image_dir, flags=cv2.IMREAD_GRAYSCALE)
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
