# SFHR-NET

The code aims to detect Parkinson's hypomimia based on facial expressions from Parkinson's desease patients and health control subjects.

The main method is as follows:

1. The folder  ./dataset/patient/  and  ./dataset/normal/  is original dataset of this task.  
2. video_segment.py will segment a complete video about 10s into some video segments containing facial expressions. 
3. Then,  crop_face.py  utilizes MTCNN algorithm to capture faces in segments.  
4.  dense_optical_flow.py  generates the optical flow from faces segments to record the motion information.  
