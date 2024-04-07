## 支持算法
- main.py
  - forward
  - gradcam/layercam/gradcam++/xgradcam
  - deepdream
- adver.py
  - fgsm
  - ifgsm
## 代码示例
- 可视化
  - 对`cow.jpg`进行forward可视化，可视化单一通道(第2通道)，不显示检测框
    ```
    python main.py --vmethod forward --spath outputs/objectDetection/yolov5/forward --ifile cow --is_channel --channel 2
    ```
  - 对`bear.jpg`进行gradcam可视化，可视化所有通道的平均值，显示检测框
    ```
    python main.py --vmethod gradcam --spath outputs/objectDetection/yolov5/gradcam --show_box --ifile bear
    ```
  - 推荐使用黑背景`zero`进行deepdream可视化，可视化所有通道的平均值
    ```
    python main.py --vmethod deepdream --spath=outputs/objectDetection/yolov5/deepdream --ifile=zero 
    ```
- 样本攻击
  -  fgsm样本攻击
    ```
    python adver.py --amethod fgsm --ifile dog --spath githubs/yolov5/data/images/fgsm
    ```
  -  ifgsm样本攻击，迭代3次，默认是2次
    ```
    python adver.py --amethod ifgsm --ifile dog --spath githubs/yolov5/data/images/test/ifgsm --iter 3
    ```

