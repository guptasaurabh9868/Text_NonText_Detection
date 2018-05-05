from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg



def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f


def ctpn(sess, net, image_name):
    global true_text,true_non_text,false_text,false_non_text
    base_name = image_name.split('/')[-1]
    label_name = image_name.split('/')[-2]
    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    print(len(boxes))
    with open('boxes.txt', 'w') as f:
        f.write(str(len(boxes)))
    if len(boxes)>0:
        if(label_name=='non_text'):
            false_non_text+=1
        else:
            true_text+=1
            cv2.imwrite(os.path.join('data/results/text',base_name),img)
    else:
        if(label_name=='text'):
            false_text+=1
        else:
            true_non_text+=1
            cv2.imwrite(os.path.join('data/results/non_text',base_name),img)


true_text=0
true_non_text=0
false_text=0
false_non_text=0

if __name__ == '__main__':
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")
    os.makedirs("data/results/text")
    os.makedirs("data/results/non_text")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    '''im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)'''
    print('done checking')

    '''im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo/text', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo/text', '*.jpg')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo/non_text', '*.jpg')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo/non_text', '*.jpg'))

    print(im_names)'''
    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', 'test.jpg'))
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)
    '''accuracy = ((true_text+true_non_text)/(true_text+false_text+false_non_text+true_non_text))
    precision = (true_text/(true_text+false_text))
    recall = (true_text/(true_text+false_non_text))
    f1_score = (2*(recall * precision)/(recall + precision))
    print('Accuracy: ' + str(accuracy))
    print ('Precision: '+ str(precision))
    print('Recall: '+str(recall))
    print('F1 Score: '+str(f1_score))'''

