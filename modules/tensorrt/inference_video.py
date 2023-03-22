
import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import argparse
from glob import glob
import os

# w = './YOLO-TensorRT8/crowd_dynamic.trt'
# w = '../onnxs/crowdhuman/crowd_640_origin_dynamic.trt'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--trt-path', help='trt-path',default='yolov7-tiny-nms.trt')
    parser.add_argument('--video-path', type=str, default='') 
    parser.add_argument('--img-size', default=640)
    parser.add_argument('--batch-size', default=1)
    opt = parser.parse_args()

    # w = '../onnxs/crowdhuman/crowd_640_dynamic_nms.trt'
    w = opt.trt_path
    bs = opt.batch_size
    img_size = opt.img_size
    video_path = opt.video_path

    assert video_path != '' 
    
    
    # print(imgfiles)

    device = torch.device('cuda:0')

    # Infer TensorRT Engine
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    context = model.create_execution_context()

    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def postprocess(boxes,r,dwdh):
        dwdh = torch.tensor(dwdh*2).to(boxes.device)
        boxes -= dwdh
        boxes /= r
        return boxes.clip_(0,6400)

    def getBindings(model,context,shape=(1,3,640,640)):
        context.set_binding_shape(0, shape)
        bindings = OrderedDict() # 데이터 입력된 순서대로 출력되는 사전
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr')) 
        
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(context.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        return bindings
    
    names = ['person','head']

    # names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    #          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    #          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    #          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    #          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    #          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    #          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
    #          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
    #          'hair drier', 'toothbrush']

    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
    origin_RGB = []

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('results.avi',fourcc,fps,(width,height))
    fps = 0


    t1 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ## 전처리
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame
        origin_RGB = []
        resize_data = []
        origin_RGB.append(img)
        image = img.copy()
        image, ratio, dwdh = letterbox(image,new_shape=(img_size,img_size), auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        resize_data.append((im,ratio,dwdh))


        np_batch = np.concatenate([data[0] for data in resize_data])
        np_batch.shape

        batch = torch.from_numpy(np_batch[0:bs]).to(device)/255
        bindings = getBindings(model,context,(bs,3,img_size,img_size))

        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        binding_addrs['images'] = int(batch.data_ptr())

        context.execute_v2(list(binding_addrs.values()))


        # show batch 32 output the first 6 pictures
        nums = bindings['num_dets'].data
        boxes = bindings['det_boxes'].data
        scores = bindings['det_scores'].data
        classes = bindings['det_classes'].data


        for batch,(num,box,score,cls) in enumerate(zip(nums.flatten(),boxes,scores,classes)):
            print(nums,box[:3],score[:3],cls[:3])
            # if batch>6:
            #     break
            RGB = origin_RGB[batch]
            ratio,dwdh = resize_data[batch][1:]
            box = postprocess(box[:num].clone(),ratio,dwdh).round().int()
            
            for idx,(b,s,c) in enumerate(zip(box,score,cls)):
                b,s,c = b.tolist(),round(float(s),3),int(c)
                name = names[c]
                color = colors[name]
                name += ' ' + str(s)
                cv2.rectangle(RGB,b[:2],b[2:],color,2)
                cv2.putText(RGB,name ,(b[0], b[1] - 2) ,cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)


        cv2.imshow('frame', RGB)
        out.write(RGB)
        t1 = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    

