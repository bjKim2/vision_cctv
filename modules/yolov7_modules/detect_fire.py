import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np 

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# from utils.general import box_iou

def bbox_iou_interbox(box1, box2, x1y1x2y2=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_x = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
    inter_y = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    inter = inter_x * inter_y
    
    inter_bbox = [torch.max(b1_x1, b2_x1),torch.max(b1_y1, b2_y1),torch.min(b1_x2, b2_x2),torch.min(b1_y2, b2_y2)]


    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    return iou,inter_bbox  # IoU



filtered_coord = []

def img_similarity(img1,img2):
    # // 이미지 읽어오기
    imgs = []
    imgs.append(img1)
    imgs.append(img2)

    hists = []
    for img in imgs:
        # // BGR 이미지를 HSV 이미지로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # // 히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # // 정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        # // hists 리스트에 저장
        hists.append(hist)

    # // 1번째 이미지를 원본으로 지정
    query = hists[0]

    # // 비교 알고리즘의 이름들을 리스트에 저장
    # methods = ['CORREL', 'CHISQR', 'INTERSECT', 'BHATTACHARYYA', 'EMD']
    methods = ['CHISQR']

    for index, name in enumerate(methods):
        # // 비교 알고리즘 이름 출력(문자열 포맷팅 및 탭 적용)
        # print('%-10s' % name, end = '\t')  
        
        # // 2회 반복(2장의 이미지에 대해 비교 연산 적용)
        for i, histogram in enumerate(hists):
            ret = cv2.compareHist(query, histogram, index) 
            
            if index == cv2.HISTCMP_INTERSECT:                   #// 교차 분석인 경우 
                ret = ret/np.sum(query)                          #// 원본으로 나누어 1로 정규화
            # print("img%d :%7.2f"% (i+1 , ret), end='\t')#.        // 비교 결과 출력
            if i == 1:
                return ret



def fire_filter(det,im0):
    
    global filtered_coord

    coord = []

    if len(filtered_coord) == 0:
        for *xyxy, conf, cls in det:
            xyxy = [int(k) for k in xyxy]
            cropped_img = im0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            filtered_coord.append([xyxy,cls,cropped_img])
    else:
        print(det.shape)
        i = 0
        for *xyxy, conf, cls in (det):

            print(xyxy)
            xyxy = [int(k) for k in xyxy]
            cropped_img = im0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            width = xyxy[2] - xyxy[0] 
            height = xyxy[3] - xyxy[1] 
            cx = (xyxy[2] + xyxy[0])//2
            cy = (xyxy[3] + xyxy[1])//2
            

            hp_cr = 0.12
            coord_range = [cx - hp_cr * width, cx + hp_cr * width,cy - hp_cr * height, cx + hp_cr * height] # min x, max x, min y, max y

            hp_area = 0.15
            area = width * height
            area_range = [(1-hp_area) * area , (1 + hp_area) * area]
            j = 0
            for *xyxy2, cls2,im02 in (filtered_coord):
                xyxy2 = [int(k) for k in xyxy2[0]]
                # print(f'i, j, len(filtered_coord) : {i}, {j}, {len(filtered_coord)}')
                # print(xyxy2,cls2)

                width2 = xyxy2[2] - xyxy2[0] 
                height2 = xyxy2[3] - xyxy2[1] 
                area2 = width2 * height2
                # hp_cr = 0.12
                # print(area_range, area2)
                if  (coord_test(coord_range,xyxy2) & (area_range[0] <= area2) & (area_range[1] >= area2)):
                    iou, inter_bbox = bbox_iou_interbox(torch.tensor(xyxy),torch.tensor(xyxy2))
                    img_sim = img_similarity(im0,im02)
                    print(f'iou : {iou} ,img_similarity(im0,im02) : {img_sim}', '#'*10)
                    if (iou >= 0.5) & (img_sim <= 0.5):
                        
                        print('same detect', '@' * 10)
                        filtered_coord[j] = [xyxy,cls,cropped_img]
                        det[i,5] = 9
                        break
                    else:
                        if j == (len(filtered_coord) -1):
                            filtered_coord.append([xyxy,cls,im0])
                            break
                else:
                    if j == (len(filtered_coord) -1):
                        filtered_coord.append([xyxy,cls,im0])
                        break
                j +=1
            i +=1
    return det
## 2가지 문제점, 일단 불이 너무 다른 좌표와 크기로 잡혀서 리스트가 커짐, 




##################### straight extract ###################

def straight_filter(det,im0):


    for *xyxy,conf,cls in det:
        xyxy = [int(k) for k in xyxy]
        src = im0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

        dst = src.copy()
        h, w = src.shape[:2]
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 100, 200 )
        edges = cv2.Canny(gray, 100, 200 )
        # 허프 선 검출, 직선으로 판단할 최소한의 점은 130개로 지정 ---②
        lines = cv2.HoughLines(edges, 1, np.pi/180, int(max(h,w) * 0.55))
        # canny = cv2.Canny(gray, 100, 200, apertureSize = 5, L2gradient = True)
        # lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 150, srn = 100, stn = 200, min_theta = 0, max_theta = np.pi)
        if lines is not None:
            for i in lines:
                rho, theta = i[0][0], i[0][1]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a*rho, b*rho

                scale = src.shape[0] + src.shape[1]

                x1 = int(x0 + scale * -b)
                y1 = int(y0 + scale * a)
                x2 = int(x0 - scale * -b)
                y2 = int(y0 - scale * a)

                x1 += xyxy[0]
                x2 += xyxy[2]
                y1 += xyxy[1]
                y2 += xyxy[3]

                print((x1, y1), (x2, y2))

                cv2.line(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)
    return im0




        
def coord_test(coord_range,xyxy):
    cx = (xyxy[2] + xyxy[0])//2
    cy = (xyxy[3] + xyxy[1])//2
    if ((coord_range[0] <= cx) & (coord_range[1] >= cx) & (coord_range[2] <= cy) & (coord_range[3] >= cy)):
        return True
    else:
        return False




def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # print('img.shape :', img.shape)
        # print('im0s.shape :', im0s.shape)
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # det = fire_filter(det,im0)
                im0 = straight_filter(det,im0)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
