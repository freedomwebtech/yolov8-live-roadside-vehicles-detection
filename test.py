import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
model=YOLO('yolov8s.pt')

stream = CamGear(source='https://www.youtube.com/watch?v=9bFOCNOarrA', stream_mode = True, logging=True).start() # YouTube Video URL as input

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

area=[(284,140),(243,153),(465,222),(524,187)]


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0



while True:    
    frame = stream.read()   
    count += 1
    if count % 3 != 0:
        continue


    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    list1=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            if result>=0:
               cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
               cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2) 
               list1.append(cy)      
   
    counter=len(list1)
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,0,255),2)
    cvzone.putTextRect(frame,f'Counter:-{counter}',(50,60),2,2)

    cv2.imshow("RGB", frame)
    

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


