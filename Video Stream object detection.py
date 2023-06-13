#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import matplotlib.pyplot as plt


# In[11]:


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph .pb'


# In[12]:


model = cv2.dnn_DetectionModel(frozen_model,config_file)


# In[13]:


classLabels = []
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


# In[14]:


print(classLabels)


# In[15]:


print(len(classLabels))


# In[16]:


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


# In[23]:


cap = cv2.VideoCapture("Video.mp4")

if not cap.isOpened():
    cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise IOError("Cannot open video")

font_scale = 3 
font = cv2.FONT_HERSHEY_PLAIN

while True: 
    ret,frame = cap.read()

    ClassIndex, confidence, bbox = model.detect(frame,confThreshold=0.55)

    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame, boxes, (255, 0,0), 2)
                cv2.putText(frame, classLabels[ClassInd-1],  (boxes[0]+10,boxes[1]+40), font, fontScale=font_scale, color=(255, 0, 0), thickness = 1)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
        
cap.release() 
cv2.destroyAllWindows()


# In[ ]:




