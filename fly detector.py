import serial
import time
import numpy as np
import cv2
#import onnxruntime as ort
import torch
from models.yolo import DetectionModel
import sys
from pathlib import Path


sys.path.append(r'C:\Users\anson\OneDrive\Documents\Senior Seminar VS Code\yolov5')
torch.serialization.add_safe_globals([DetectionModel])

arduinoData = serial.Serial("COM6",9600)
#onnx_model_path = "C:/Users/anson/OneDrive/Documents/Senior Seminar VS Code/yolov5/runs/train/exp4/weights/best.onnx"
#ort_session = ort.InferenceSession(onnx_model_path)
torch_model = r"C:\Users\anson\OneDrive\Documents\Senior Seminar VS Code\yolov5\runs\train\exp4\weights\best.pt"
model = torch.load(torch_model, weights_only=False)['model'].float()
model.eval()

def send_coordinates_to_arduino(x,y,w,h):
    #Convert coorindates to a string and send it to Arduino
    coordinates = f"{x},{y}\r"
    arduinoData.write(coordinates.encode())
    print(f"x{x}Y{y}\n")

def preprocess_frame(frame, input_size=640):
    # Resize while maintaining aspect ratio
   # Preprocess the image: resize and convert to tensor
    img = cv2.resize(frame, (input_size, 640))  # Resize to YOLOv5 input size
    img = img[:, :, ::-1]  # Convert BGR to RGB
    img = torch.from_numpy(img.copy()).float()  # Convert to tensor and make a copy
    img /= 255.0  # Normalize to [0, 1]
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)  # Add batch dimension

    return img



file_location = r"C:\Users\\anson\Downloads\\new flies footage.mp4"
capture = cv2.VideoCapture(file_location)


while True:
    isTrue, frame = capture.read()

    img = preprocess_frame(frame)
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(img)
   
    
    #outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: img})
    for i, det in enumerate(pred):
         if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                #This is what I changed
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])),(int(xyxy[2]),int(xyxy[3]))
                    center_point = round((c1[0]+c2[0])/2), round((c1[1]+c2[1]/2))
                    center_points = center_points
                    circle = cv2.circle(im0,center_point,5,(0,255,0),2)
                    text_coord = cv2.putText(im0,str(center_point),center_point,cv2.FONT_HERSHEY_PLAIN,2,(0,255,255)) 
"""
    for det in outputs[0]:
        x1, y1, x2, y2, conf, cls = det[:6]
        
        x1 = int(x1.item()) if x1.numel() == 1 else int(x1[0].item())
        y1 = int(y1.item()) if y1.numel() == 1 else int(y1[0].item())
        x2 = int(x2.item()) if x2.numel() == 1 else int(x2[0].item())
        y2 = int(y2.item()) if y2.numel() == 1 else int(y2[0].item())

        w, h = x2 - x1, y2 - y1

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Send coordinates to Arduino
        send_coordinates_to_arduino(x1, y1, w, h)

"""
 

    


    cv2.imshow('YOLOv5 Object Detection',frame)
    
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()