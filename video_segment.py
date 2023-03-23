import cv2,os

def video_seg(path):
    fps = 64
    size = (1920,1080)
    video = list(filter(lambda x: x.find('.MOV')!=-1, os.listdir(path)))
    for videoname in video:
        print(videoname)
        m = 0
        while True:
            i = 0
            videoWriter =cv2.VideoWriter('./dataset/video_segment2/'+videoname[:-4]+'Seg_'+str(m)+'.avi',cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)
            videoCapture = cv2.VideoCapture(path+videoname)
            while True:
                success,frame = videoCapture.read()
                if success:
                    i += 1
#                     print('i = ',i)
                    if(i>=m and i < m+fps):
                        videoWriter.write(frame)
                else:
                    print(m,'end')   
                    break
            if m>300:
                break 
            else:
                m +=32

path_normal = './dataset/normal/'
path_patient = './dataset/patient/'
# video_seg(path_normal)
video_seg(path_patient)
video_seg(path_normal)
