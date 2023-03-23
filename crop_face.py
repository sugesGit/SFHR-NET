from mtcnn.mtcnn import MTCNN
from PIL import Image
import cv2,numpy as np,os

class MTCNN_face_detection():

    def __init__(self,path,videoname,savepath):
        self.videoname = videoname
        self.pic_path = path+self.videoname
        self.savepath = savepath
    
    def video_read(self):
        videocapture = cv2.VideoCapture(self.pic_path)
        i = 0
        while True:
            success,frame = videocapture.read()
            if success:
                i += 1
                print('i = ',i)
                self.face_detection(frame,i)
            else:
                print('end')   
                break

    def face_detection(self,img,img_id):
        # img = Image.open('imageNew.jpg') # 打开当前路径图像
        # print(np.array(img).shape)
        print('img.shape:',img.shape)

        detector = MTCNN()
        face_list = detector.detect_faces(img) # face detect and alignment
        print(face_list)
        
        for face in face_list:
            box = face["box"]    
            x,y,w,h = box
            print('box:',box,img.shape,img_id)
#             cv2.rectangle(img,(round(x*0.9),round(y*0.8)),(round(x+w*1.2),round(y+h*1.1)),(255,255,255),2)
            img = img[round(y*0.8):round(y+h*1.1),round(x*0.9):round(x+w*1.2)]
            img = cv2.resize(img,(570,650))
#         cv2.imwrite("./dataset/Untitled Folder/result.jpg",img)
        
        if img_id<10:
            cv2.imwrite(self.savepath+self.videoname[:-4]+'_0'+str(img_id)+"frame.jpg",img)
        else:
            cv2.imwrite(self.savepath+self.videoname[:-4]+'_'+str(img_id)+"frame.jpg",img)

face_path = './dataset/crop_face2/'
video_path = './dataset/video_segment2/'
video_segments = filter(lambda x : x.find('.avi')!=-1,os.listdir(video_path))
mylog = open('./dataset/logs/crop_face2.log','w')
errorlog = open('./dataset/logs/non_crop_face2.log','w')
for video_segment in video_segments:
    os.mkdir(face_path+video_segment[:-4]+'/')
    mtcnn_face = MTCNN_face_detection(video_path,video_segment,face_path+video_segment[:-4]+'/')
    try:
        mtcnn_face.video_read()
        print ('video_segment:',video_segment,file=mylog)
    except:
        print ('error_video_segment:',video_segment,file=errorlog)
        pass
mylog.close()
errorlog.close()
