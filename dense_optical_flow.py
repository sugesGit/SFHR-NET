import cv2,numpy as np,os

def optical(face_path,face_seg):
    base_path = os.path.join(face_path,face_seg)
    frames = os.listdir(base_path)
    frames.sort()
    frame0 = cv2.imread(base_path+'/'+frames[0])
    prvs = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame0)
    hsv[...,1] = 255
    i = 2
    os.mkdir('./dataset/optical_flow/'+face_seg+'/')
    for frame in frames[1:]:
        try:
            img = cv2.imread(base_path+'/'+frame)
            print(base_path+'/'+frame)
            print(img.shape)
        except:
            pass
        next = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        print(i)

    #         cv2.imwrite('./opticao flow/'+face_seg[:-4]+'/opticalfb'+str(i)+'frame.png',frame2)
        if i < 10:
            cv2.imwrite('./dataset/optical_flow/'+face_seg+'/opticalhsv0'+str(i)+'frame.png',rgb)
        else:   
            cv2.imwrite('./dataset/optical_flow/'+face_seg+'/opticalhsv'+str(i)+'frame.png',rgb)

        prvs = next
        i += 1
        

face_path = './dataset/crop_face/'
face_segs = filter(lambda x :x.find('IMG')!=-1,os.listdir(face_path))
for face_seg in face_segs:
    optical(face_path,face_seg)
