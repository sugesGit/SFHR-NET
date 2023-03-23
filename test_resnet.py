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
#         transforms.CenterCrop(224),
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
    predicted = torch.max(out.data, 1).indices
    print(predicted)
    return predicted

def mainfunction(model,use_gpu,sourcepath):    
    filedir = os.listdir(sourcepath)
    correct = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for picfile in filedir:
        pics = os.listdir(sourcepath+picfile)
        pics.sort()
        i = 0
        print(picfile)
        judge = 0
        if int(picfile[4:8])<800:
            judge = 1
        for pic in pics:
#             print('i',i)
            i += 1
            if i%2==0:
                continue
#             print(sourcepath+picfile+'/'+pic)
            if i == 1:
                videoseg = sumfigure(sourcepath+picfile+'/'+pic)
            elif i == 3:
                videoseg = torch.cat((videoseg[None],sumfigure(sourcepath+picfile+'/'+pic)[None]),0)
            else:
                videoseg = torch.cat((videoseg,sumfigure(sourcepath+picfile+'/'+pic)[None]),0)
            if i>=65:
                break
        print(videoseg.shape)
        pre = predictor(videoseg, model, use_gpu)
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
        # if judge==pre:
        #     correct += 1
        print('judge:',judge,'pre',pre,'correct',correct, 'TP,FP,FN,TN', TP,FP,FN,TN)
    # precision = correct/len(filedir)
    try:
        precision = TP/(TP+FP)
    except:
        precision = 0
    recall = TP/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)
    specificity = TN/(TN+FP)
    accuracy = correct/len(filedir)
    return precision,recall,F1,accuracy,specificity 
    # return precision

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()
modelpath = './weights/temp2/'
sourcepath = './dataset/crop_face/Testset/'
modeldir = os.listdir(modelpath)
modeldir.sort()
print(modeldir)
modelnames = filter(lambda x:x.find('.th')!=-1,modeldir)
mylog = open('./test_logs/spatial_resnet.log','a')
for modelname in modelnames:
    print(modelname) 
    # model = generate_model(model_depth=101).cuda()
    model = VGGV()
    model = model.cuda()
    model.load_state_dict(torch.load(modelpath+modelname))
    precision,recall,F1,accuracy,specificity = mainfunction(model,use_gpu,sourcepath)

    print('**********')
    print('model:',modelname)
    print('precision:',precision)
    print('recall:',recall)
    print('F1:',F1)
    print('accuracy:',accuracy)
    print('model:',modelname, ':precision:',precision, ':recall:',recall, ':F1:',F1, ':accuracy:',accuracy, ':specificity:',specificity, file = mylog)

    # print('**********')
    # print('model:',modelname)
    # print('precision:',precision)
    # print('**********',file = mylog)
    # print('model:',modelname,file = mylog)
    # print('precision:',precision, file = mylog)
    mylog.flush()
mylog.close()
