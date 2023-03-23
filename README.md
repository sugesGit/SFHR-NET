# SFHR-NET

The code aims to detect Parkinson's hypomimia based on facial expressions from Parkinson's desease patients and health control subjects.
The paper is available [[here]](https://dl.acm.org/doi/abs/10.1145/3476778).

The overview can be seen as follow:
![the overview can be seen as follow](https://github.com/ronronnersu/SFHR-NET/blob/main/figure/overview.png)

## Usage

1. The folder  ./dataset/patient/  and  ./dataset/normal/  is original dataset of this task  
2. Segmenting a complete video about 10s into some video segments containing facial expressions 
```
python3 video_segment.py
```
3. Using MTCNN algorithm to capture faces in segments  
```
python3 crop_face.py
```
4. Generateing the optical flow from faces segments to record the motion information
```
python3 dense_optical_flow.py
```
5. Training spatial features
```
python3 ordered_space_train.py
```
6. Training temporal features
```
python3 ordered_temporal_train.py
```
7. Testing integrated features combining spatial features with temporal features
```
python3 test.py
```
8. Generating visualized maps for possible lesions
```
python3 attention_map.py
```

## Experiments
| Backbone  | Precision(\%) | Recall(\%) | F1-score(\%) | Accuracy(\%) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| VGG | 90.99 | 100.00 | 95.28 | 94.15 |
| SFHR-NET(ours)  | 100.00 | 98.99 | 99.49 | 99.39 |


## Visualized results
![the visualized result can be seen as follow](https://github.com/ronronnersu/SFHR-NET/blob/main/figure/attention_map.png)
