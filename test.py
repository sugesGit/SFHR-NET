#添加精度，召回率，F1指标的计算
import torch
import torch.nn as nn
import os, glob, numpy as np
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
from model.VGGV import VGGV
from model.ResNet import generate_model

def sumfigure(img_path):
    transform = transforms.Compose([
        transforms.CenterCrop((648,570)),
        transforms.Resize((216,190)),
        transforms.ToTensor()]
    )
 
    img = Image.open(img_path)
    img = transform(img)
    return img
 
def predictor(videoseg, net, use_gpu):
    videoseg = videoseg.permute(1,0,2,3)
    x = Variable(torch.unsqueeze(videoseg, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    out = net(x)
    predicted = out.cpu().data.numpy()
#     predicted = torch.max(out.data, 1).indices
    return predicted

def sum_data(sourcepath,picfile):
    pics = os.listdir(sourcepath+picfile)
    pics.sort()
    i = 0
    for pic in pics:
            i += 1
            if i%2==0:
                continue
            if i == 1:
                videoseg = sumfigure(sourcepath+picfile+'/'+pic)
            elif i == 3:
                videoseg = torch.cat((videoseg[None],sumfigure(sourcepath+picfile+'/'+pic)[None]),0)
            else:
                videoseg = torch.cat((videoseg,sumfigure(sourcepath+picfile+'/'+pic)[None]),0)
            if i>=64:
                break
    return videoseg

def mainfunction(model_s,model_t,use_gpu,sourcepath_spatial,sourcepath_temporal):    
    filedir = os.listdir(sourcepath_spatial)
    correct = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for picfile in filedir:
        judge = 0
        if int(picfile[4:8])<800:
            judge = 1
        videoseg_s = sum_data(sourcepath_spatial,picfile)
        videoseg_t = sum_data(sourcepath_temporal,picfile)
        pre_spatical = predictor(videoseg_s, model_s, use_gpu)
        pre_temporal = predictor(videoseg_t, model_t, use_gpu)
        pre = 0.8*pre_spatical+0.2*pre_temporal
        pre = np.argmax(pre)
        if judge==pre:
            correct += 1
            if judge == 1:
                TP += 1
            else:
                TN += 1
        elif judge == 1:
            FN += 1
        else:
            FP += 1
        print('judge:',judge,'pre',pre,'correct',correct,'TP,FP,FN,TN',TP,FP,FN,TN, pre_spatical, pre_temporal, 0.01*pre_spatical+0.99*pre_temporal)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)
    specificity = TN/(TN+FP)
    accuracy = correct/len(filedir)
    return precision,recall,F1,accuracy,specificity 

def capture_modelname(modelpath,sourcepath):
    modeldir = list(filter(lambda x: x.find('model')!=-1, os.listdir(modelpath)))
    modeldir.sort()
    return modeldir

def predict(modelpath_spatial,model_spaticial,sourcepath_spatial,modelpath_temporal,model_temporal,sourcepath_temporal):
    # model_s = generate_model(model_depth=101).cuda()
    model_s = VGGV()
    model_s = model_s.cuda()
    model_s.load_state_dict(torch.load(modelpath_spatial+model_spaticial))
    # model_t = generate_model(model_depth=101).cuda()
    model_t = VGGV()
    model_t = model_t.cuda()
    model_t.load_state_dict(torch.load(modelpath_temporal+model_temporal))
    precision,recall,F1,accuracy,specificity = mainfunction(model_s,model_t,use_gpu,sourcepath_spatial,sourcepath_temporal)
    return precision,recall,F1,accuracy,specificity

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()

sourcepath_spatial = './dataset/crop_face/Testset/'
sourcepath_temporal = './dataset/optical_flow/Testset/'

modelpath_spatial = './weights/saved_weights/ours/spatial_weights/'
model_spaticial = capture_modelname(modelpath_spatial,sourcepath_spatial)
modelpath_temporal = './weights/saved_weights/ours/temporal_weights/'
model_temporal = capture_modelname(modelpath_temporal,sourcepath_temporal)

mylog = open('./test_logs/test.log','a')
for index in range(len(model_spaticial)):
    print(model_spaticial[index],model_temporal[index])
    precision,recall,F1,accuracy,specificity = predict(modelpath_spatial,model_spaticial[index],\
                                sourcepath_spatial,modelpath_temporal,model_temporal[index],sourcepath_temporal)
    print('**********')
    print('model:',model_spaticial[index])
    print('precision:',precision)
    print('recall:',recall)
    print('F1:',F1)
    print('accuracy:',accuracy)
    print('model:',model_spaticial[index],':precision:',precision,':recall:',recall,':F1:',F1,':accuracy:',accuracy,':specificity:',specificity,file = mylog)
#     print('precision:',precision, file = mylog)
#     print('recall:',recall, file = mylog)
#     print('F1:',F1, file = mylog)
#     print('accuracy:',accuracy, file = mylog)
    mylog.flush()
mylog.close()
