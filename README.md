- 使用fgsm对样本进行白盒攻击
  ```
  python adver.py --smethod fgsm --ifile dog --apath githubs/yolov5/data/images/test/fgsm
  ```
- 使用可视化模块进行前向可视化
  ```
  python main.py --smethod forward --spath outputs/objectDetection/yolov5/test/forward --ifile cow --is_channel
  ```
