- 样本攻击
  -  fgsm样本攻击
    ```
    python adver.py --smethod fgsm --ifile dog --apath githubs/yolov5/data/images/test/fgsm
    ```
- 可视化
  - forward可视化，只显示特定通道
    ```
    python main.py --smethod forward --spath outputs/objectDetection/yolov5/test/forward --ifile cow --is_channel
    ```
