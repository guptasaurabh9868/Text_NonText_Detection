# Text_NonText_Detection
Machine Learning Project For Course
Android Code: https://github.com/guptasaurabh9868/androidTextDetection

The android code have been uploaded in different repository.
The paper implemented here is https://arxiv.org/abs/1609.03605

Some code has been taken from https://github.com/eragonruan/text-detection-ctpn. The VGG16 model will be trained using vGG16_train_cnn in Image classification Text Non-text and saves the model their itself. The generated model will inturn uses the generated model to train the overall CTPN architecture and saves the final model. 

For training, run python3 Image\ Classification\ Text-NonText/ train.py
For testing, run python3 Image\ Classification\ Text-NonText/ test.py

Setup
requirements: tensorflow1.3, cython0.24, opencv-python, easydict,(recommend to install Anaconda)

Dataset used:
The benchmark used for training and testing can be found here http://mclab.eic.hust.edu.cn/~xbai/textDis/textDis.html

All the best to everyone who uses it and let me know, if you have any query. ;)
