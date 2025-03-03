import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import argparse
import pathlib
from pathlib import Path
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import yolov5
import numpy as np
import matplotlib.pyplot as plt

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
class TestDataset(Dataset):
    def __init__(self, img_dir, img_size=640):
        self.img_dir = img_dir
        self.img_size = (640,640)
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])
        origin_image = cv2.imread(img_name)
        #图像预处理
        im,_,dwh =letterbox(origin_image, new_shape=self.img_size,stride=stride,auto=False)
        #print(im.shape)
        # HWC to CHW, BGR to RGB
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im=im.transpose(2,0,1) #HWC->CHW
        im=im.astype(np.float32)
        im = np.ascontiguousarray(im)  # 确保图像内存连续
        im=im/255
        #归一化
        """
        for i in range(3):
            image[:,:,i]=(image[:,:,i]-self.mean[i])/(self.std[i])
        """
        
        return torch.tensor(im), img_name,dwh,origin_image  # 返回张量和图像名称,dwh为填充的wh

def scale_boxes(img1_shape, boxes, img0_shape):
    """将边界框从图像1的形状缩放到图像0的形状。

    Args:
        img1_shape (tuple): 调整后的图像形状 (width, height)。
        boxes (numpy.ndarray): 输入的边界框，格式为 [x1, y1, x2, y2, conf, cls]。
        img0_shape (tuple): 原始图像形状 (width,height )。

    Returns:
        numpy.ndarray: 缩放后的边界框。
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 计算缩放比例
    pad = (
        (img1_shape[0] - img0_shape[0] * gain) / 2, 
        (img1_shape[1] - img0_shape[1] * gain) / 2
    )  # 计算填充

    # 仅在边界框的相应坐标上应用填充和缩放
    boxes[..., [0, 2]] -= pad[0]  # 调整 x 坐标
    boxes[..., [1, 3]] -= pad[1]  # 调整 y 坐标
    boxes[..., :4] /= gain  # 缩放边界框坐标

    # 限制边界框在图像内
    clip_boxes(boxes, img0_shape)
    
    return boxes

def clip_boxes(boxes, shape):
    """裁剪边界框坐标到图像形状范围内。

    Args:
        boxes (numpy.ndarray): 边界框坐标数组。
        shape (tuple): 图像形状 (height, width)。
    """
    if isinstance(boxes, torch.Tensor):  # 如果是Torch Tensor
        boxes[..., 0].clamp_(0, shape[0])  # x1
        boxes[..., 1].clamp_(0, shape[1])  # y1
        boxes[..., 2].clamp_(0, shape[0])  # x2
        boxes[..., 3].clamp_(0, shape[1])  # y2
    else:  # 如果是Numpy array
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[0])  # x1, x2
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[1])  # y1, y2

def iou(box1, box2):
    """计算两个边界框的 IoU。"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0


def nms(predictions, conf_threshold, iou_threshold):
    """非极大值抑制（NMS）实现。"""
    filtered_boxes = []
    predictions = predictions[predictions[:, 4] >= conf_threshold]  # 置信度过滤

    # 按置信度降序排序
    sorted_indices = np.argsort(predictions[:, 4])[::-1]
    predictions = predictions[sorted_indices]

    while len(predictions) > 0:
        current_box = predictions[0]
        filtered_boxes.append(current_box)
        remaining_boxes = []

        for box in predictions[1:]:
            if iou(current_box[:4], box[:4]) < iou_threshold:
                remaining_boxes.append(box)

        predictions = np.array(remaining_boxes)

    return np.array(filtered_boxes)




def visualize_predictions(model, dataloader, device, num_images=5, conf_threshold=0.25,iou_thres=0.45,output_path_test=r"D:\develop\output",origin_size=(1080,2376)):
    model.eval()  # 设置模型为评估模式
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

    with torch.no_grad():  
        for i, (images, img_names,dwh,origin_images) in enumerate(dataloader):
            if i >= num_images:
                break
            origin_images = origin_images.to(device)
            origin_image=origin_images[0].cpu().numpy()
            images = images.to(device)
            outputs = model(images)
            image = images[0].cpu().numpy().transpose(1, 2, 0)  # CHW 转 HWC
            predictions=outputs[0].boxes.data.cpu().numpy()
            # 应用置信度过滤和 IoU 过滤
            predictions = nms(predictions, conf_threshold, iou_thres)
            h, w, _ = image.shape
            scale_predictions=scale_boxes((w,h),predictions,origin_size)
            #print(scale_predictions[0][:4])
            image = image*255.0
            image = image.astype(np.uint8)
            #image = np.ascontiguousarray(image) # 确保图像内存连续
            #还原成原图大小
            image=cv2.resize(image,origin_size)
            #画框
            for index,pred in enumerate(scale_predictions):
                x1, y1, x2, y2, conf, cls = pred
                color=[int(c*255) for c in colors[int(cls)]]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(origin_image, (x1, y1), (x2, y2),color, 2)
                #cv2.putText(origin_image, f'{names[int(cls)]}:{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
                cv2.putText(origin_image, f'{index+1}', (x1, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
                        #保存图片
            img_name = os.path.basename(img_names[0])
            output_path = os.path.join(output_path_test, img_name)
            cv2.imwrite(output_path, origin_image)
    print(f"预测结果已保存到 {output_path_test}")

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型权重路径，例如：./ui_area_dev1.pt')
    parser.add_argument('--img_dir', type=str, required=True, help='测试图片存储路径，例如：./images')
    parser.add_argument('--output_path', type=str, required=True, help='输出路径，例如：./output')
    parser.add_argument('--img_size', type=int, default=640, help='输入图像大小，默认640')
    parser.add_argument('--origin_size', type=int, nargs=2, default=(1080, 2376), help='原始图像尺寸，格式为：宽 高')
    parser.add_argument('--conf', type=float, default=0.4, help='置信度阈值，默认0.4')
    parser.add_argument('--iou', type=float, default=0.1, help='IoU 阈值，默认0.1')


    args = parser.parse_args()

    # 加载 YOLOv8模型
    from ultralytics import YOLO
    model=YOLO("D:\develop\mywork\icon_detect\model.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    stride, names= model.stride, model.names

    # 获取类别数量
    num_classes = len(names)
    # 为每一个类别分配一种颜色
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    # 加载测试数据集
    test_dataset = TestDataset(args.img_dir, img_size=args.img_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 标注预测结果
    visualize_predictions(model, test_dataloader, device, num_images=len(test_dataset), output_path_test=args.output_path, origin_size=args.origin_size)
