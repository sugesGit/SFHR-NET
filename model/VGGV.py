import torch
import torch.nn as tnn
import numpy as np
from torch.autograd import Variable

class VGGV(tnn.Module):
    def __init__(self):
        super(VGGV, self).__init__()
        self.layer1 = tnn.Sequential(

            # 1-1 conv layer
            tnn.Conv3d(3, 64, kernel_size=3, padding=1),
            tnn.BatchNorm3d(64),
            tnn.ReLU(),

            # 1-2 conv layer
            tnn.Conv3d(64, 64, kernel_size=3, padding=1),
            tnn.BatchNorm3d(64),
            tnn.ReLU(),

            # 1 Pooling layer
            tnn.MaxPool3d(kernel_size=2, stride=2))

        self.layer2 = tnn.Sequential(

            # 2-1 conv layer
            tnn.Conv3d(64, 128, kernel_size=3, padding=1),
            tnn.BatchNorm3d(128),
            tnn.ReLU(),

            # 2-2 conv layer
            tnn.Conv3d(128, 128, kernel_size=3, padding=1),
            tnn.BatchNorm3d(128),
            tnn.ReLU(),

            # 2 Pooling lyaer
            tnn.MaxPool3d(kernel_size=2, stride=2))

        self.layer3 = tnn.Sequential(

            # 3-1 conv layer
            tnn.Conv3d(128, 256, kernel_size=3, padding=1),
            tnn.BatchNorm3d(256),
            tnn.ReLU(),

            # 3-2 conv layer
            tnn.Conv3d(256, 256, kernel_size=3, padding=1),
            tnn.BatchNorm3d(256),
            tnn.ReLU(),

            # 3-3 conv layer
            tnn.Conv3d(256, 256, kernel_size=3, padding=1),
            tnn.BatchNorm3d(256),
            tnn.ReLU(),

            # 3 Pooling layer
            tnn.MaxPool3d(kernel_size=2, stride=2))

        self.layer4 = tnn.Sequential(

            # 4-1 conv layer
            tnn.Conv3d(256, 512, kernel_size=3, padding=1),
            tnn.BatchNorm3d(512),
            tnn.ReLU(),

            # 4-2 conv layer
            tnn.Conv3d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm3d(512),
            tnn.ReLU(),

            # 4-3 conv layer
            tnn.Conv3d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm3d(512),
            tnn.ReLU(),

            # 4 Pooling layer
            tnn.MaxPool3d(kernel_size=2, stride=2))

        self.layer5 = tnn.Sequential(

            # 5-1 conv layer
            tnn.Conv3d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm3d(512),
            tnn.ReLU(),

            # 5-2 conv layer
            tnn.Conv3d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm3d(512),
            tnn.ReLU(),

            # 5-3 conv layer
            tnn.Conv3d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm3d(512),
            tnn.ReLU(),

            # 5 Pooling layer
            tnn.MaxPool3d(kernel_size=2, stride=2))

        self.layer6 = tnn.Sequential(

            # 6 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            tnn.Linear(15360, 4096),
#             tnn.BatchNorm1d(4096),
            tnn.ReLU())
        

        self.layer7 = tnn.Sequential(

            # 7 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            tnn.Linear(4096, 4096),
#             tnn.BatchNorm1d(4096),
            tnn.ReLU())

        self.layer8 = tnn.Sequential(

            # 8 output layer
            tnn.Linear(4096, 2),
#             tnn.BatchNorm1d(1000),
            tnn.Softmax())


        self.layer5_1 = (

            tnn.MaxPool3d(kernel_size=2, stride=2)

            )
        self.target_layers = 8
        self.gradients = []

    def forward(self, x):
        # print('forward1')
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        vgg16_features = out.view(out.size(0), -1)
        out = self.layer6(vgg16_features)
        out = self.layer7(out)
        out = self.layer8(out)
        print('out1',out)
        return out

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward2(self, x):
        # print('forward2')
        self.gradients = []
        target_activations = []
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        for name, module in self.layer5._modules.items():
            x = module(x)
            if int(name) == self.target_layers:
                handle = x.register_hook(self.save_gradient)
                target_activations = x #1*512*2*13*11
                # print('---------------',target_activations.shape)
            
        vgg16_features = x.view(x.size(0), -1)
        x = self.layer6(vgg16_features)
        x = self.layer7(x)
        x = self.layer8(x)

       
        index = np.argmax(x.cpu().data.numpy())
        one_hot = np.zeros((1, x.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        one_hot = torch.sum(one_hot.cuda() * x)
         
        # print('one_hot:',one_hot)
        one_hot.backward(retain_graph=True)

        grads_val = self.gradients[-1]
        # print('grads_val[0, 1, 1, 3, :]:',grads_val[0, 1, 1, 3, :])
        grads_val[grads_val>0] = 1
        grads_val[grads_val<0] = -1
        
        # print('grads_val[0, 1, 1, 3, :]:',grads_val[0, 1, 1, 3, :])
        # print('target_activations[0, 1, 1, 3, :]:',target_activations[0, 1, 1, 3, :])
        target_activations = torch.mul(target_activations, grads_val)
        
        # print(target_activations[0, 1, 1, 3, :])


        '''
        grads_val = self.gradients[-1].cpu().data.numpy() #(1, 512, 2, 13, 11)
        print('gradients:',self.gradients[-1].shape)
        weights = np.mean(grads_val, axis = (2,3,4))[0, :] #512
        # print('weights:',weights.shape)
        print(target_activations[0].shape)
        for i, weight in enumerate(weights):
            target_activations[0][0,i,:,:,:] = target_activations[0][0,i,:,:,:] * weight
        target_activations = target_activations[0].cpu().data.numpy()
        target_activations = np.maximum(target_activations, 0)
        target_activations = torch.tensor(target_activations).cuda()
        '''

        x = self.layer5_1(target_activations)
        vgg16_features = x.view(x.size(0), -1)
        x = self.layer6(vgg16_features)
        x = self.layer7(x)
        out = self.layer8(x)
        print('out2',out)
        handle.remove()
        # del handle
        return out
