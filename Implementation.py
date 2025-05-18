import sys
import os
import cv2
import torch
import pathlib

# 关键路径修正 --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(BASE_DIR, 'yolov5')
sys.path.insert(0, YOLOV5_PATH)  # 必须放在其他导入之前

# 临时重定向 PosixPath 到 WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors
# ------------------------------------------------------------

def main():
    # 加载模型
    model = attempt_load(os.path.join(BASE_DIR, 'best.pt'), device='cpu')
    model.eval()

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 预处理
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

            # 推理
            with torch.no_grad():
                pred = model(img_tensor)[0]

            # 后处理
            pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

            # 绘制结果
            annotator = Annotator(frame)
            if pred[0] is not None:
                det = pred[0]
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls)))

            # 显示画面
            cv2.imshow('Fire Detection', annotator.result())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 恢复原始 PosixPath
    pathlib.PosixPath = temp

if __name__ == "__main__":
    main()
