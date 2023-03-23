import torch
from torch.autograd import Variable,Function
from torchvision import utils,models,transforms
from model.VGGV import VGGV
from PIL import Image
import cv2,sys,os,argparse,numpy as np

class FeatureExtractor():
    def __init__(self, layer1,layer2,layer3,layer4,layer5, target_layers):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.model = [self.layer1,self.layer2,self.layer3,self.layer4,self.layer5]
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)
        # print('gradients:',self.gradients[0].shape)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        key = 0
        for layer in self.model:
            key += 1
            for name, module in layer._modules.items():
                x = module(x)
                # print(module)
                if key == 5 and name in self.target_layers[1]:
                    # print('-----------------------')
                    x.register_hook(self.save_gradient)
                    # print(self.gradients)
                    outputs += [x]
        return outputs, x

class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.layer1,self.model.layer2,\
                                                  self.model.layer3,self.model.layer4,self.model.layer5,target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        # print('target_activations, feature map:',target_activations[0].shape, output.shape)
        output = output.view(output.size(0), -1)
        output = self.model.layer6(output)
        output = self.model.layer7(output)
        output = self.model.layer8(output)
#       feature map:target_activations=1,512,1,14,14; classification result:output=1,2
        return target_activations, output

def preprocess_image(videoseg):
    videoseg = videoseg.permute(1,0,2,3)
    x = Variable(torch.unsqueeze(videoseg, dim=0).float(), requires_grad=False)
    return x

def show_cam_on_image(img, mask,savepath):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(savepath, np.uint8(255 * cam))

def show_cam_on_image2(img, mask,savepath):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap
    cam = cam / np.max(cam)
    cv2.imwrite(savepath, np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)


    def __call__(self, input, index = None):
        
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            # print('output:', output)
            # print('features.shape, index:', features[0].shape, index)

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        # print('one_hot:', one_hot.cuda() * output)

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        self.model.layer1.zero_grad()
        self.model.layer2.zero_grad()
        self.model.layer3.zero_grad()
        self.model.layer4.zero_grad()
        self.model.layer5.zero_grad()
        self.model.layer6.zero_grad()
        self.model.layer7.zero_grad()
        self.model.layer8.zero_grad()
        # print('--------------1-----------------------------')
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        print('grad:', grads_val.shape)
        # print('--------------2----------------------------')
        
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        # print('grad:', grads_val.shape,grads_val[0,1,0,2])
        # print('--------------2----------------------------')
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        # print('grad:', grads_val.shape,grads_val[0,1,0,2])
        
        target = features[-1]
        # print(target.shape)
        target = target.cpu().data.numpy()[0, :]       #512*2*13*11
        target = np.mean(target, axis = (1))
        target = np.expand_dims(target, axis = 1)
        # print('target==features:',target.shape)
        weights = np.mean(grads_val, axis = (2,3,4))[0, :] #512
        # print('weights:',weights.shape)
        # print(weights)
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, 0, :, :]

        # print(cam[0,1,:])
        cam = np.maximum(cam, 0)
        # print(cam[0,1,:])
        # print(cam.shape)
        cam = cv2.resize(cam[0], (190,216))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam,index


def get_args():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")
    return use_cuda

def sumfigure(img_path):

    transform = transforms.Compose([
                transforms.CenterCrop((648,570)),
                transforms.Resize((216,190)),
                transforms.ToTensor()])

    img = Image.open(img_path)
    img = transform(img)

    return img

def main(sourcepath):
    filedir = os.listdir(sourcepath)
    for picfile in filedir:
        print( )
        print('the current segment:',picfile)
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
        # print('original videos:', videoseg.shape)
        input = preprocess_image(videoseg)

        target_index = None
        mask,index = grad_cam(input, target_index)
        if index:
            result = 'Parkinson'
        else:
            result = 'Health'
        transform = transforms.Compose([
                                        transforms.CenterCrop((648,570)),
                                        transforms.Resize((216,190))
                                        ])
        img_path = sourcepath+picfile+'/'+picfile+'_01frame.jpg'
        img = Image.open(img_path)
        img = transform(img)
        img = np.float32(img) / 255
        # print('img.shape:',img.shape,'mask.shape:',mask.shape)
        # savepath = './attention_map/train/'+picfile+'_'+result+'.jpg'\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        savepath = './attention_map2/test/'+picfile+'_'+result+'.jpg'
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # print(savepath)
        show_cam_on_image2(img, mask, savepath)
    

if __name__ == '__main__':
    use_cuda = get_args()
    model = VGGV()
    model = model.cuda()
    # modelpath = './weights/spatial_weights3/spacial_model40.th'\\\\\\\\\\\\\\\\\\\\
    modelpath = './backup/weights/spatial_weights/spacial_model.th'
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    model.load_state_dict(torch.load(modelpath))
    grad_cam = GradCam(model,target_layer_names = ['layer5',"8"], use_cuda=use_cuda)
    
    # sourcepath = './dataset/crop_face/Trainset/'\\\\\\\\\\\\\\\\\
    sourcepath = './dataset/crop_face/Testset/'
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    videoseg = main(sourcepath)
    
    print('finished!')
