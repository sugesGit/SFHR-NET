import torch
import torch.nn as nn
import os, glob, numpy as np
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
from model.VGGV import VGGV

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
    predicted = torch.max(out.data, 1).indices
    print(predicted)
    return predicted

def mainfunction(model,use_gpu,sourcepath):    
    filedir = os.listdir(sourcepath)
    correct = 0
    for picfile in filedir:
        pics = os.listdir(sourcepath+picfile)
        pics.sort()
        i = 0
        print(picfile)
        judge = 0
        if int(picfile[4:8])<800:
            judge = 1
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
            if i>=65:
                break
        print(videoseg.shape)
        pre = predictor(videoseg, model, use_gpu)
        if judge==pre:
            correct += 1
        print('judge:',judge,'pre',pre,'correct',correct)
    precision = correct/len(filedir)
    return precision

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()
modelpath = './weights/saved_weights/ours/temporal_weights/'
sourcepath = './dataset/optical_flow/Testset/'
modeldir = os.listdir(modelpath)
modeldir.sort()
print(modeldir)
modelnames = filter(lambda x:x.find('.th')!=-1,modeldir)
mylog = open('./test_logs/resting_temporal.log','w')
for modelname in modelnames:
    print(modelname)
    model = VGGV()
    model = model.cuda()
    model.load_state_dict(torch.load(modelpath+modelname))
    precision = mainfunction(model,use_gpu,sourcepath)
    print('**********')
    print('model:',modelname)
    print('precision:',precision)
    print('**********',file = mylog)
    print('model:',modelname,file = mylog)
    print('precision:',precision, file = mylog)
    mylog.flush()
mylog.close()
