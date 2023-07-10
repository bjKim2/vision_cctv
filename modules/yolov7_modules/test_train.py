import os

if __name__ == '__main__':
    cmd = 'python train.py --workers 12 --device 0 --batch-size 32 --data ./Safetyvest/data.yaml --img 416 416 --cfg cfg/training/yolov7.yaml --weights \'\' --name yolov7 --hyp data/hyp.scratch.p6.yaml --epochs 1500'
    print(cmd)
    os.system(cmd)