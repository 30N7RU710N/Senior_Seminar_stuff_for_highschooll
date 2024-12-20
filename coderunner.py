import subprocess       

subprocess.run(("python", "yolov5/detect.py", "--weight", "yolov5/runs/train/exp4/weights/best.pt", "--source", '0' , '--save-txt'))
