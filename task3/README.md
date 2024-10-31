# 2021302181152 邓鹏 第三次实验

## 1.模型下载

https://github.com/serengil/deepface_models/releases

文件复制到C:\Users\用户名\.deepface\weights中

## 2.代码文件介绍

### ·使用Python3+OpenCV+dlib实现人脸识别与关键点（landmarks）实时检测

`1.py` 运行代码，会利用电脑摄像头捕捉实时画面，按q退出

### ·结合实验任务1使用Python3+OpenCV+Deepface实现人脸情感检测

`2.py` 运行代码，会利用电脑摄像头捕捉实时画面检测人脸感情，每十帧一次，显示情感、年龄、性别

### ·使用Python3+dlib实现人脸伪造

`3.py` 运行 `python 3.py 2.png 1.png` 会生成output.png

### ·使用Python3+Face-X-Ray实现人脸伪造图像检测

`Face-X-Ray-2` 目录下将database目录下的1.jpg重命名为target_1.jpg复制到source目录下

运行`python faceBlending.py --srcFacePath path_to_Face-X-Ray-2\source\ --faceDatabase path_to_Face-X-Ray-2\database -t 50`

会在dump目录下生成相应文件