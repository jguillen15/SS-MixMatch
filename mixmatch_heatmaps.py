# -*- coding: utf-8 -*-
"""MixMatch_training.ipynb
"""

import argparse

parser = argparse.ArgumentParser(description = "MixMatch_training")
parser.add_argument('--K_transforms', action='store', dest='K_transforms', default=2, type=int)
parser.add_argument('--T_sharpening', action='store', dest='T_sharpening', default=0.25, type=float)
parser.add_argument('--alpha_mix', action='store', dest='alpha_mix', default=0.75, type=float)
parser.add_argument('--balanced', action='store', dest='balanced', default=-1, type=int)
parser.add_argument('--batch_size', action='store', dest='batch_size', default=10, type=int)
parser.add_argument('--epochs', action='store', dest='epochs', default=50, type=int)
parser.add_argument('--lambda_unsupervised', action='store', dest='lambda_unsupervised', default=200, type=int)
parser.add_argument('--DIR_SUMMARIES', action='store', dest='DIR_SUMMARIES', default="/home/sacalderon/Johan/summaries/", type=str)
parser.add_argument('--lr', action='store', dest='lr', default=0.0001, type=float)
parser.add_argument('--mode', action='store', dest='mode', default="partial_supervised", type=str)
parser.add_argument('--model', action='store', dest='model', default="densenet_pre", type=str)
parser.add_argument('--norm_stats', action='store', dest='norm_stats', default="auto", type=str)
parser.add_argument('--num_classes', action='store', dest='num_classes', default=2, type=int)
parser.add_argument('--number_labeled', action='store', dest='number_labeled', default=10, type=int)
parser.add_argument('--path_labeled', action='store', dest='path_labeled', default="/home/sacalderon/Johan/Hen_paper/labeled/batch_0", type=str)
parser.add_argument('--path_unlabeled', action='store', dest='path_unlabeled', default="/home/sacalderon/Johan/Hen_paper/unlabeled_reduced", type=str)
parser.add_argument('--RUN_NAME', action='store', dest='RUN_NAME', default="test", type=str)
parser.add_argument('--size_image', action='store', dest='size_image', default=224, type=int)
parser.add_argument('--desired_labeled_classes_dist', action='store', dest='desired_labeled_classes_dist', default='0.5, 0.5', type=str)
parser.add_argument('--rampup_coefficient', action='store', dest='rampup_coefficient', default=3000, type=int)
parser.add_argument('--save_weights', action='store', dest='save_weights', default=False, type=bool)
parser.add_argument('--weight_decay', action='store', dest='weight_decay', default=0.001, type=float)
parser.add_argument('--workers', action='store', dest='workers', default=10, type=int)
parser.add_argument('--DIR_MODELS', action='store', dest='DIR_MODELS', default="/home/sacalderon/Johan/models", type=str)
parser.add_argument('--batch', action='store', dest='batch', default="0", type=str)

args = parser.parse_args()

"""#Code and Imports"""

import torch
import logging
import sys
import numpy as np
import torchvision
import os
import scikitplot as skplt
import pandas as pd
import cv2
from torch.nn import functional as F
from PIL import Image as Pili

from datetime import datetime
from PIL import ImageFile
from fastai.callbacks import CSVLogger, SaveModelCallback, EarlyStoppingCallback
from torchvision.utils import save_image
from fastai.vision import *
from fastai.callbacks import CSVLogger
from numbers import Integral
from fastai.metrics import error_rate
from matplotlib import pyplot as plt
from fastai.train import ClassificationInterpretation
from fastai.callbacks.hooks import model_summary
from os import listdir
from os.path import isfile, join
from math import log, sqrt
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from fastai.callbacks.hooks import *
import scipy.ndimage


torch.cuda.set_device(0)

warnings.simplefilter("ignore", UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    #for idx in class_idx:
    cam = weight_softmax[0].dot(feature_conv.reshape((nc, h*w))) #idx
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
    
def show_cam(CAMs, width, height, orig_image, class_idx, all_classes, cont):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + orig_image * 0.5
        # put class label text on the result
        #cv2.putText(result, all_classes[class_idx[i]], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        #cv2.imshow('CAM', result/255.)
        #cv2.waitKey(0)
        scont = str(cont)
        nombre = "CAM_Heatmap_"+ scont + ".png"
        cv2.imwrite(nombre, result)
        


def print_and_log(msg):
  with open(DIR_RUN_SUMMARIES + "/log.txt", "a") as log_file:
    log_file.write(msg)
    log_file.write("\n")

def plot_ROC(y, preds, model_file_name):
  plt.clf()
  skplt.metrics.plot_roc(y, preds, plot_micro=False, plot_macro=False)
  plt.savefig(DIR_RUN_SUMMARIES + "/" + model_file_name + "_roc_plot.png")
  plt.clf()
  
def freeze_upper_layers(model):
  for name, parameter in model.named_parameters():
    if (name.split(".")[0] == "classifier"):
      continue

    parameter.requires_grad = False
   
    
class CustomCMScores(ConfusionMatrix):
    """Example of confusion matrix in self.cm:   
        tensor([[14., 66.],
                [ 2.,  3.]])

        Rows correspond to actual class and columns to predicted class,
        thus for this example TN = 14, TP = 3, FP = 66, FN = 2
    """
    def _recall(self):
        rec = torch.diag(self.cm) / self.cm.sum(dim=1)
        return (rec * Tensor([0,1])).sum()

    def _specificity(self):
        rec = torch.diag(self.cm) / self.cm.sum(dim=1)
        return (rec * Tensor([1,0])).sum()

    def _balanced_accuracy(self):
        b_accuracy = (self._recall() + self._specificity()) / 2
        return b_accuracy
    def _g_mean(self):
        g_mean = torch.sqrt(self._recall() * self._specificity())
        return g_mean

    def _h_mean(self):
        h_mean = (5 * self._recall() * self._specificity()) / ((4 * self._specificity()) + self._recall())
        return h_mean
    
class GMean(CustomCMScores):            
    def on_epoch_end(self, last_metrics, **kwargs): 
        return add_metrics(last_metrics, self._g_mean())
class Specificity(CustomCMScores):
    "Computes the Specificity."
    def on_epoch_end(self, last_metrics, **kwargs): 
        return add_metrics(last_metrics, self._specificity())

class BalancedAccuracy(CustomCMScores):
    "Computes the Balanced Accuracy."
    def on_epoch_end(self, last_metrics, **kwargs): 
        return add_metrics(last_metrics, self._balanced_accuracy())

#F2 score using specificity instead of precision
class HMean(CustomCMScores):
  def on_epoch_end(self, last_metrics, **kwargs): 
        return add_metrics(last_metrics, self._h_mean())
    
    
#============Metrics.py code=========================================================================================================
def calculate_mean_std(path_dataset):
    """
    Calculate mean and std of dataset
    :param path_dataset:
    :return:
    """
    path_dataset = path_dataset + "/train"  # Para que solo calcule la normalizacion con los datos de train
    dataset = torchvision.datasets.ImageFolder(path_dataset, transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor() ]))
    #print(dataset)
    return get_mean_and_std(dataset)

def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset.
    :param dataset:
    :return:
    """
    data_loader = torch.utils.data.DataLoader(dataset,  num_workers= 5, pin_memory=True, batch_size =1)

    #init the mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    k = 1
    for inputs, targets in data_loader:
        #mean and std from the image
        #print("Processing image: ", k)
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
        k += 1

    #normalize
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print("mean: " + str(mean))
    print("std: " + str(std))
    return mean, std

#============Metricscode===========================================================================================================
def get_file_names_in_path(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles


def pil2fast(img, im_size):
    data_transform = transforms.Compose(
        [transforms.ToTensor()])
    return Image(data_transform(img))


def measure_model_accuracy_test(learner, img_path, class_label, im_size, ssdl):
    list_images = get_file_names_in_path(img_path)
    print("total of images ", len(list_images))
    num_test = len(list_images)
    correct_preds = 0
    wrong_preds = 0

    list_predictions = []

    for i in range(0, num_test):
        complete_path = img_path + list_images[i]
        image_pil = Pili.open(complete_path).convert('RGB')
        image_fastai = pil2fast(image_pil, im_size=im_size)
        cat_tensor, tensor_class, model_output = learner.predict(image_fastai, with_dropout=False)
        if (tensor_class.item() == class_label):
            correct_preds += 1
        else:
            wrong_preds += 1

        if (ssdl):
            model_output = nn.functional.softmax(model_output, dim=0)

        list_predictions += [model_output.tolist()]    

    accuracy = correct_preds/num_test
    return accuracy, correct_preds, wrong_preds, list_predictions



def get_precision_recall_f1_score(fastai_model, test_image_path_c0, test_image_path_c1, im_size, ssdl):
    #0 is normal or no pathology
    acc_c0, correct_preds_c0, wrong_preds_c0, preds0 = measure_model_accuracy_test(fastai_model, test_image_path_c0,
                                                                          class_label=0,
                                                                          im_size=im_size, ssdl=ssdl)
    print("get_precision_recall_f1_score correct preds c0 ",  correct_preds_c0, " wrong preds c0 ", wrong_preds_c0)
    # 1 is covid-19 positive
    acc_c1, correct_preds_c1, wrong_preds_c1, preds1 = measure_model_accuracy_test(fastai_model, test_image_path_c1,
                                                                           class_label=1,
                                                                           im_size=im_size, ssdl=ssdl)
    print("get_precision_recall_f1_score correct preds c1 ", correct_preds_c1, " wrong preds c1 ", wrong_preds_c1)

    list_predictions = np.asarray(preds0 + preds1)
    list_labels = [0] * len(preds0) + [1] * len(preds1)
    
    total_accuracy = (correct_preds_c0 + correct_preds_c1) / (correct_preds_c0 + correct_preds_c1 + wrong_preds_c0 + wrong_preds_c1)
    true_positives = correct_preds_c1
    false_positives = wrong_preds_c0
    false_negatives = wrong_preds_c1
    
    recall = true_positives / (true_positives + false_negatives)
    
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = "n/a"
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = "n/a"
    
    if(recall == "n/a" or precision == "n/a" or (recall + precision) == 0):
        f1_score = "n/a"
        f2_score = "n/a"
    else:
        f1_score = (2 * recall * precision) / (precision + recall)
        f2_score = (5 * recall * precision) / ((4 * precision) + recall)
        
    specificity = correct_preds_c0 / (correct_preds_c0 + false_positives) #True negatives / (true negatives + false positives
    balanced_accuracy = (recall + specificity) / 2
    g_mean = sqrt(recall * specificity)

    auroc = roc_auc_score(list_labels, list_predictions[:,1])
    
    return f1_score, recall, precision, total_accuracy, specificity, balanced_accuracy, g_mean, auroc, f2_score


def create_classification_metrics_summary(summary_name, f1_score, recall, precision, accuracy, specificity, balanced_accuracy, g_mean, auroc, f2_score):
    items = [["f1",f1_score], ["recall",recall], ["precision",precision], ["accuracy",accuracy], ["specificity", specificity], ["balanced_accuracy", balanced_accuracy], ["g_mean", g_mean], ["auroc", auroc], ["f2_score", f2_score]]
    df = pd.DataFrame(items)
    df.to_csv(summary_name, index=False)


def test_model_f1_score_fully_supervised(learner, batch_number, path_data, dataset_name, id_model):
    class_label_iod_test_data = 0
    img_path_iod_c0 = path_data + "/test/" + str(class_label_iod_test_data) + "/"

    class_label_iod_test_data = 1
    img_path_iod_c1 = path_data + "/test/" + str(class_label_iod_test_data) + "/"
    
    f1_score_no_ssdl, recall_no_ssdl, precision_no_ssdl, accuracy_no_ssdl, specificity_no_ssdl, balanced_accuracy_no_ssdl, g_mean_no_ssdl, auroc_no_ssdl, f2_score_no_ssdl = get_precision_recall_f1_score(learner, img_path_iod_c0, img_path_iod_c1, im_size=IMAGE_SIZE, ssdl=False)
    


    summaries_path_container = DIR_RUN_SUMMARIES + "/summaries_classification"
    summaries_path = summaries_path_container + "/batch_" + str(batch_number) + "_" + str(id_model)
    if not os.path.exists(summaries_path):
      if not os.path.exists(summaries_path_container):
        os.mkdir(summaries_path_container)
      os.mkdir(summaries_path)

    
    print("No SSDL f1 score: ", f1_score_no_ssdl, " recall: ", recall_no_ssdl, " precision: ", precision_no_ssdl, " accuracy: ", accuracy_no_ssdl, " specificity: ", specificity_no_ssdl, " balanced_accuracy: ", balanced_accuracy_no_ssdl, " g_mean: ", g_mean_no_ssdl, " auroc: ", auroc_no_ssdl, " f2_score: ", f2_score_no_ssdl)
    summary_name = summaries_path + "/F1_SCORE_SUMMARY_BATCH_NO_SSDL.csv"
    create_classification_metrics_summary(summary_name, f1_score_no_ssdl, recall_no_ssdl, precision_no_ssdl, accuracy_no_ssdl, specificity_no_ssdl, balanced_accuracy_no_ssdl, g_mean_no_ssdl, auroc_no_ssdl, f2_score_no_ssdl)


def test_model_f1_score(learner, batch_number, path_data, dataset_name, id_model, ssdl=False):
    class_label_iod_test_data = 0
    path_labeled = path_data + "/"
    
    img_path_iod_c0 = path_data + "/test/" + str(
        class_label_iod_test_data) + "/"
    class_label_iod_test_data = 1
    img_path_iod_c1 = path_data + "/test/" + str(
        class_label_iod_test_data) + "/"

    summaries_path_container = DIR_RUN_SUMMARIES + "/summaries_classification"
    summaries_path = summaries_path_container + "/batch_" + str(batch_number) + "_" + str(id_model)
    if not os.path.exists(summaries_path):
      if not os.path.exists(summaries_path_container):
        os.mkdir(summaries_path_container)
      os.mkdir(summaries_path)

    if (ssdl):

      model_name_ssdl = 'SSDL_model_batch_' + str(batch_number) + "_" + str(id_model)      
      learner_ssdl = learner
      f1_score_ssdl, recall_ssdl, precision_ssdl, accuracy_ssdl, specificity_ssdl, balanced_accuracy_ssdl, g_mean_ssdl, auroc_ssdl, f2_score_ssdl = get_precision_recall_f1_score(learner_ssdl,
                                                                                          img_path_iod_c0,
                                                                                          img_path_iod_c1, im_size=IMAGE_SIZE, ssdl=ssdl)
      summary_name = summaries_path + "/F1_SCORE_SUMMARY_BATCH_SSDL.csv"
      print("SSDL f1 score: ", f1_score_ssdl, " recall: ", recall_ssdl, " precision: ", precision_ssdl, " accuracy: ", accuracy_ssdl, " specificity: ", specificity_ssdl, " balanced_accuracy: ", balanced_accuracy_ssdl, " g_mean: ", g_mean_ssdl, " auroc: ", auroc_ssdl, " f2_score: ", f2_score_ssdl)
      create_classification_metrics_summary(summary_name, f1_score_ssdl, recall_ssdl, precision_ssdl, accuracy_ssdl, specificity_ssdl, balanced_accuracy_ssdl, g_mean_ssdl, auroc_ssdl, f2_score_ssdl)
      
      
    else:
      model_name_no_ssdl = 'NO_SSDL_model_batch_' + str(batch_number) + "_" + str(id_model)
      learner_no_ssdl = learner
      f1_score_no_ssdl, recall_no_ssdl, precision_no_ssdl, accuracy_no_ssdl, specificity_no_ssdl, balanced_accuracy_no_ssdl, g_mean_no_ssdl, auroc_no_ssdl, f2_score_no_ssdl = get_precision_recall_f1_score(learner_no_ssdl, img_path_iod_c0, img_path_iod_c1, im_size=IMAGE_SIZE, ssdl=ssdl)
      
      print("No SSDL f1 score: ", f1_score_no_ssdl, " recall: ", recall_no_ssdl, " precision: ", precision_no_ssdl, " accuracy: ", accuracy_no_ssdl, " specificity: ", specificity_no_ssdl, " balanced_accuracy: ", balanced_accuracy_no_ssdl, " g_mean: ", g_mean_no_ssdl, " auroc: ", auroc_no_ssdl, " f2_score: ", f2_score_no_ssdl)
      summary_name = summaries_path + "/F1_SCORE_SUMMARY_BATCH_NO_SSDL.csv"
      create_classification_metrics_summary(summary_name, f1_score_no_ssdl, recall_no_ssdl, precision_no_ssdl, accuracy_no_ssdl, specificity_no_ssdl, balanced_accuracy_no_ssdl, g_mean_no_ssdl, auroc_no_ssdl, f2_score_no_ssdl)
    

def calculate_metrics(learner, type_model, path_data, dataset_name, model_id):
  batch_number = int(path_data.split("/")[-1].split("_")[-1])
  if (type_model == "semi_supervised"):
      test_model_f1_score(learner=learner, batch_number = batch_number, path_data=path_data, dataset_name=dataset_name, id_model=model_id, ssdl=True)
          
  if (type_model == "partial_supervised"):
      test_model_f1_score(learner=learner, batch_number = batch_number, path_data=path_data, dataset_name=dataset_name, id_model=model_id, ssdl=False)
          
  if (type_model == "fully_supervised"):
      test_model_f1_score_fully_supervised(learner=learner, batch_number = batch_number, path_data=path_data, dataset_name=dataset_name, id_model=model_id)

#============MixMatch code=========================================================================================================    

def calculate_weights(list_labels):
    """
    Calculate the class weights according to the number of observations
    :param list_labels:
    :return:
    """    
    print("Using balanced loss: " + str(BALANCED))
    print_and_log("Using balanced loss: " + str(BALANCED))
    array_labels = np.array(list_labels)
    list_classes = np.unique(array_labels)
    weight_classes = np.zeros(len(list_classes))
    for curr_class in list_classes:

        number_observations_class = len(array_labels[array_labels == curr_class])
        print("Number observations " + str(number_observations_class) + " for class " + str(curr_class))
        print_and_log("Number observations " + str(number_observations_class) + " for class " + str(curr_class))
        weight_classes[curr_class] = 1 / number_observations_class

    weight_classes = weight_classes / weight_classes.sum()
    weight_classes_tensor = torch.tensor(weight_classes, device ="cuda:0" )
    print("Weights to use: " + str(weight_classes))
    print_and_log("Weights to use: " + str(weight_classes))
    return weight_classes_tensor

class MultiTransformLabelList(LabelList):
    def __getitem__(self, idxs: Union[int, np.ndarray]) -> 'LabelList':
        """
        Create K transformed images for the unlabeled data
        :param idxs:
        :return:
        """
        "return a single (x, y) if `idxs` is an integer or a new `LabelList` object if `idxs` is a range."
        
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            if self.item is None: 
                x,y = self.x[idxs], self.y[idxs]
            else:                 
                x,y = self.item, 0
            if self.tfms or self.tfmargs:
                x = [x.apply_tfms(self.tfms, **self.tfmargs) for _ in range(K_VALUE)]
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve':False})
            if y is None: y=0
            return x,y
        else: 
            return self.new(self.x[idxs], self.y[idxs])
        
#I'll also need to change the default collate function to accomodate multiple augments
def MixmatchCollate(batch):
    batch = to_data(batch)
    if isinstance(batch[0][0],list):
        batch = [[torch.stack(s[0]),s[1]] for s in batch]
    return torch.utils.data.dataloader.default_collate(batch)

class MixMatchImageList(ImageList):
    def filter_train(self, num_items, seed=23488):
        train_idxs = np.array([i for i,o in enumerate(self.items) if Path(o).parts[-3] != "test"])
        valid_idxs = np.array([i for i,o in enumerate(self.items) if Path(o).parts[-3] == "test"])
        np.random.seed(seed)
        keep_idxs = np.random.choice(train_idxs,num_items,replace=False)
        self.items = np.array([o for i,o in enumerate(self.items) if i in np.concatenate([keep_idxs,valid_idxs])])
        print("Number of labeled observations: " + str(len(keep_idxs)))
        print("First labeled id: " + str(keep_idxs[0]))
        print("Number of validation observations: " + str(len(valid_idxs)))
        print("Number of training observations " + str(len(train_idxs)))
        print_and_log("Number of labeled observations: " + str(len(keep_idxs)))
        print_and_log("First labeled id: " + str(keep_idxs[0]))
        print_and_log("Number of validation observations: " + str(len(valid_idxs)))
        print_and_log("Number of training observations " + str(len(train_idxs)))

        return self
    def filter_train_balance_control(self, num_items, path_labeled, path_unlabeled, seed=23488, desired_proportions = []):
        """
        :param num_items:
        :param seed:
        :param desired_proportions: The desired percentage of observations per class, to control class umbalance for labeled observations
        :return:
        """
        global class_weights
        # for reproducibility
        np.random.seed(seed)
        num_items_per_class = []
        #create a list of lists with the items per class
        items_per_class = [[] for _ in range(len(desired_proportions))]

        #calculate the number of items per class
        for i in range(0, len(desired_proportions)):
            num_items_per_class += [int(desired_proportions[i] * num_items) ]
        #get label dictionary
        label_dictionary = self.get_labels_dict()
        print(label_dictionary)
        if (path_unlabeled == ""):
            path_unlabeled = path_labeled
        # this means that a customized unlabeled dataset is not to be used, just pick the rest of the labelled data as unlabelled
        if (path_unlabeled == path_labeled):
            train_idxs_all_list = []
            for i, observation in enumerate(self.items):
                if (Path(observation).parts[-3] != "test"):
                    path_1 = str(Path(observation))
                    train_idxs_all_list += [i]
                    substr_train = re.findall(r"/\d+/", path_1)
                    label_num_str = re.findall(r"\d+", substr_train[0])
                    label = int(label_num_str[0])
                    proxy_label = label_dictionary[label]
                    # add the element to the corresponding sub list of observations for this class, according to label
                    items_per_class[proxy_label] += [i]
                    
        else:
            # IGNORE THE DATA ALREADY IN THE UNLABELED DATASET
            dataset_unlabeled = torchvision.datasets.ImageFolder(path_unlabeled + "/train/")
            list_file_names_unlabeled = dataset_unlabeled.imgs
            for i in range(0, len(list_file_names_unlabeled)):
                # delete root of path
                # print("Before ", list_file_names_unlabeled[i])
                list_file_names_unlabeled[i] = list_file_names_unlabeled[i][0].replace(path_unlabeled, "")
                # print("after ", list_file_names_unlabeled[i])
            list_train = []
            # add  to train if is not in the unlabeled dataset

            for i, observation in enumerate(self.items):
                path_1 = str(Path(observation))
                sub_str = path_labeled
                path_2 = path_1.replace(sub_str, "")
                path_2 = path_2.replace("train/", "")
                is_path_in_unlabeled = path_2 in list_file_names_unlabeled
                # add the observation to the train list, if is not in the unlabeled dataset
                if (not "test" in path_2 and not is_path_in_unlabeled):
                    list_train += [i]
                    #get substring with train and class folder
                    substr_train = re.findall(r"/\d+/", path_2)
                    label_num_str = re.findall(r"\d+", substr_train[0])
                    label = int(label_num_str[0])
                    proxy_label = label_dictionary[label]
                    #add the element to the corresponding sub list of observations for this class, according to label
                    items_per_class[proxy_label] += [i]
            
        #concat all the observations
        keep_idxs_all = []
        
        for i in range(0, len(desired_proportions)):
            #for each class, select the given number of random labels
            items_class_i = items_per_class[i]
            keep_idxs_i = np.random.choice(np.array(items_class_i), num_items_per_class[i], replace=False)
            keep_idxs_all += keep_idxs_i.tolist()

        keep_idxs_all_np = np.array(keep_idxs_all)

        #the test dataset is done when building the folder
        valid_idxs = np.array([i for i, observation in enumerate(self.items) if Path(observation).parts[-3] == "test"])
        print("Number of labeled observations: " + str(len(keep_idxs_all_np)))
        print("First labeled id: " + str(keep_idxs_all_np[0]))
        print("Number of  validation observations: " + str(len(valid_idxs)))
        print("Number  of training observations " + str(len(keep_idxs_all_np)))
        print_and_log("Number of labeled observations: " + str(len(keep_idxs_all_np)))
        print_and_log("First labeled id: " + str(keep_idxs_all_np[0]))
        print_and_log("Number of  validation observations: " + str(len(valid_idxs)))
        print_and_log("Number  of training observations " + str(len(keep_idxs_all_np)))
        self.items = np.array([o for i, o in enumerate(self.items) if i in np.concatenate([keep_idxs_all_np, valid_idxs])])
        return self
    def get_labels_dict(self):
        """
        Get the dictionary with the labels
        :return:
        """
        proxy_label_counter = 0
        dictionary = {-1:0}
        
        for i, observation in enumerate(self.items):
            if (Path(observation).parts[-3] != "test"):
                path_1 = str(Path(observation))

                substr_train = re.findall(r"/\d+/", path_1)
                label_num_str = re.findall(r"\d+", substr_train[0])
                label = int(label_num_str[0])
                #if the element does not exist, add it
                try:
                    a = dictionary[label]
                except:
                    dictionary[label] = proxy_label_counter
                    proxy_label_counter += 1
        
        return dictionary

def mixup(a_x, a_y, b_x, b_y, alpha=0.75):
    
    l = np.random.beta(ALPHA_VALUE, ALPHA_VALUE)
    l = max(l,1-l)
    x = l * a_x + (1-l) * b_x
    y = l* a_y + (1-l) * b_y
    return x,y

def sharpen(p, T=0.5):
    u = p ** (1/T_VALUE)
    return u / u.sum(dim=1,keepdim=True)

# Mixmatch algorithm

class MixupLoss(nn.Module):
    def forward(self, preds, target, unsort=None, ramp=None, bs=None):
        """
        Ramp, unsort and bs is None when doing validation
        :param preds:
        :param target:
        :param unsort:
        :param ramp:
        :param bs:
        :return:
        """

        if(BALANCED==5):
            return self.forward_balanced_cross_entropy(preds, target, unsort, ramp, bs)
        else:

            #assign the same weight for the classes, in disregard with the dataset
            weight = 1 / len(class_weights)
            for i in range(0, len(class_weights)):
                class_weights[i] = weight
            #class_weights = torch.tensor([0.3333, 0.3333, 0.3333], device ="cuda:0")
            return self.forward_balanced_cross_entropy(preds, target, unsort, ramp, bs)

    def forward_cross_entropy(self, preds, target, unsort=None, ramp=None, bs=None):

        if unsort is None:
            return F.cross_entropy(preds, target)

        calculate_cross_entropy = nn.CrossEntropyLoss()
        preds = preds[unsort]
        preds_l = preds[:bs]
        preds_ul = preds[bs:]
        # calculate log of softmax, to ensure correct usage of cross entropy
        # one column per class, one batch per  row
        # preds_l = torch.log_softmax(preds_l,dim=1)

        preds_ul = torch.softmax(preds_ul, dim=1)
        # TARGETS CANNOT BE 1-K ONE HOT VECTOR
        (highest_values, highest_classes) = torch.max(target[:bs], 1)

        highest_classes = highest_classes.long()

        loss_x = calculate_cross_entropy(preds_l, highest_classes)
        # loss_x = -(preds_l * target[:bs]).sum(dim=1).mean()
        loss_u = F.smooth_l1_loss(preds_ul, target[bs:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        return loss_x + LAMBDA_VALUE * ramp * loss_u


    def forward_original(self, preds, target, unsort=None, ramp=None, num_labeled=None):
        
        """
        Implements the forward pass of the loss function
        :param preds: predictions of the model
        :param target: ground truth targets
        :param unsort: ?
        :param ramp: ramp weight
        :param num_labeled:
        :return:
        """
        if unsort is None:
            #used for evaluation
            return F.cross_entropy(preds,target)
        preds = preds[unsort]
        #labeled and unlabeled observations were packed in the same array
        preds_l = preds[:num_labeled]
        preds_ul = preds[num_labeled:]
        #apply logarithm to softmax of output, to ensure the correct usage of cross entropy
        preds_l = torch.log_softmax(preds_l,dim=1)
        preds_ul = torch.softmax(preds_ul,dim=1)
        #consider using CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        loss_x = -(preds_l * target[:num_labeled]).sum(dim=1).mean()
        loss_u = F.mse_loss(preds_ul, target[num_labeled:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        return loss_x + LAMBDA_VALUE * ramp * loss_u

    

    def forward_balanced_cross_entropy(self, preds, target, unsort=None, ramp=None, bs=None):
        global class_weights
        if unsort is None:
            return F.cross_entropy(preds, target)
        #weights_labeled = self.get_weights_observations(target[:bs]).float()
        #class_weights = torch.tensor(torch.tensor([0.3333, 0.3333, 0.3333]), device="cuda:0")


        weights_unlabeled = self.get_weights_observations(target[bs:]).float()
        
        #CHANGE 1!!a
        calculate_cross_entropy = nn.CrossEntropyLoss(weight = class_weights.float())
        #calculate_cross_entropy = nn.CrossEntropyLoss()
        preds = preds[unsort]
        preds_l = preds[:bs]
        preds_ul = preds[bs:]
        # calculate logs of softmax, to ensure correct usage of cross entropy
        # one column per class, one batch per row x
        # preds_l = torch.log_softmax(preds_l,dim=1)
        preds_ul = torch.softmax(preds_ul, dim=1)
        # TARGETS CANNOT BE 1-K ONE HOT VECTOR
        (highest_values, highest_classes) = torch.max(target[:bs], 1)
        highest_classes = highest_classes.long()
        loss_x = calculate_cross_entropy(preds_l, highest_classes)
        # loss_x = -(preds_l * target[:bs]).sum(dim=1).mean()
        #CHANGE 2!!
        #loss_u = F.smooth_l1_loss(weights_unlabeled *preds_ul, weights_unlabeled *target[bs:])
        loss_u = F.mse_loss(weights_unlabeled * preds_ul, weights_unlabeled * target[bs:])
        #loss_u = F.mse_loss(preds_ul, target[bs:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        #CHANGE 3!!!
        #args.lambda_unsupervised = 200
        #print(args.lambda_unsupervised)
        return loss_x + LAMBDA_VALUE * ramp * loss_u

    def get_weights_observations(self, array_predictions):
        global class_weights
        # class_weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        # each column is a class, each row an observation
        num_classes = array_predictions.shape[1]
        num_observations = array_predictions.shape[0]
        (highest_values, highest_classes) = torch.max(array_predictions, 1)
        # turn the highest_classes array a column vector
        highest_classes_col = highest_classes.view(-1, 1)
        # highest classes for all the observations (rows) and classes (columns)
        highest_classes_all = highest_classes_col.repeat(1, num_classes)
        # print("highest classes all")
        # print(highest_classes_all)
        # scores all
        scores_all = class_weights[highest_classes_all]
        scores_all.to(device="cuda:0")
        return scores_all

class MixMatchTrainer(LearnerCallback):
    _order=-20
    def on_train_begin(self, **kwargs):
        self.l_dl = iter(data_labeled.train_dl)
        self.smoothL, self.smoothUL = SmoothenValue(0.98), SmoothenValue(0.98)
        self.it = 0
        
    def on_batch_begin(self, train, last_input, last_target, **kwargs):
        if not train: return
        try:
            x_l,y_l = next(self.l_dl)
        except:
            self.l_dl = iter(data_labeled.train_dl)
            x_l,y_l = next(self.l_dl)
            
        x_ul = last_input
        
        with torch.no_grad():
            ul_labels = sharpen(torch.softmax(torch.stack([self.learn.model(x_ul[:,i]) for i in range(x_ul.shape[1])],dim=1),dim=2).mean(dim=1))
            
        x_ul = torch.cat([x for x in x_ul])
        ul_labels = torch.cat([y.unsqueeze(0).expand(K_VALUE,-1) for y in ul_labels])
        
        l_labels = torch.eye(data_labeled.c).cuda()[y_l]
        
        w_x = torch.cat([x_l, x_ul])
        w_y = torch.cat([l_labels, ul_labels])
        idxs = torch.randperm(w_x.shape[0])
        
        mixed_input, mixed_target = mixup(w_x, w_y, w_x[idxs],w_y[idxs])
        bn_idxs = torch.randperm(mixed_input.shape[0])
        unsort = [0] * len(bn_idxs)
        for i,j in enumerate(bn_idxs): unsort[j] = i
        mixed_input = mixed_input[bn_idxs]
    

        ramp = self.it / rampup_coefficient if self.it < rampup_coefficient else 1.0
        return {"last_input": mixed_input, "last_target": (mixed_target, unsort, ramp, x_l.shape[0])}
    
    def on_batch_end(self, train, **kwargs):
        if not train: return
        self.smoothL.add_value(self.learn.loss_func.loss_x)
        self.smoothUL.add_value(self.learn.loss_func.loss_u)
        self.it += 1

    

def get_dataset_stats(dataset, path_labeled, path_unlabeled, mode):
    if(dataset == "MNIST"):
        # stats for MNIST, replace!!
        meanDatasetComplete = [0.1307, 0.1307, 0.1307]
        stdDatasetComplete = [0.3081, 0.3081, 0.3081]

    elif (dataset == "imagenet"):
        # normalization values for pretrained torch models
        meanDatasetComplete = [0.485, 0.456, 0.406]
        stdDatasetComplete = [0.229, 0.224, 0.225]

    elif (dataset == "auto"):
        (meanDatasetComplete, stdDatasetComplete) = calculate_mean_std(path_labeled)

    elif (dataset == "Hen"):
        meanDatasetComplete = [0.408924, 0.378666, 0.356273]
        stdDatasetComplete = [0.200269, 0.20207, 0.207837]
        
        

    if(mode == "semi_supervised" and path_unlabeled  != ""):
        (meanDatasetComplete_unlabeled, stdDatasetComplete_unlabeled) = calculate_mean_std(path_unlabeled)
    else:
        (meanDatasetComplete_unlabeled, stdDatasetComplete_unlabeled) = (meanDatasetComplete, stdDatasetComplete)
    
    return (meanDatasetComplete, stdDatasetComplete, meanDatasetComplete_unlabeled, stdDatasetComplete_unlabeled)

"""#Data and Training"""

def load_data(dataset_name, mode, path_labeled, path_unlabeled = "", desired_labeled_classes_dist = [0.5, 0.5]):
  global data_labeled, class_weights
  
  

  #get dataset mean and std
  (meanDatasetComplete_labeled, stdDatasetComplete_labeled, meanDatasetComplete_unlabeled, stdDatasetComplete_unlabeled) = get_dataset_stats(dataset_name, path_labeled, path_unlabeled, mode)
  
  
  if (path_unlabeled == ""):
      path_unlabeled = path_labeled
  
  print("Loading labeled data from: " + path_labeled)
  print_and_log("Loading labeled data from: " + path_labeled)
  print("Loading unlabeled data from: " + path_unlabeled)
  print_and_log("Loading unlabeled data from: " + path_unlabeled)
  
  data_labeled = (MixMatchImageList.from_folder(path_labeled,  presort=True)
                  .filter_train_balance_control(NUMBER_LABELED_OBSERVATIONS, path_labeled, path_unlabeled, seed = 4200, desired_proportions = desired_labeled_classes_dist)
                  .split_by_folder(valid="test") #test on all 10000 images in test set
                  .label_from_folder()
                  .transform(get_transforms(do_flip = True, flip_vert = True, max_zoom=1, max_warp=None, p_affine=0, p_lighting = 0),
                               size=IMAGE_SIZE)
                  #On windows, must set num_workers=0. Otherwise, remove the argument for a potential performance improvement
                  .databunch(bs=BATCH_SIZE, num_workers=WORKERS)
                  .normalize((meanDatasetComplete_labeled, stdDatasetComplete_labeled)))

  print("Data labeled loading ...")
  print_and_log("Data labeled loading ...")

  # Train set ids
  train_set = set(data_labeled.train_ds.x.items)

  list_labels = data_labeled.train_ds.y.items
  print(train_set)
  print(list_labels)
  print_and_log(str(train_set))
  print_and_log(str(list_labels))


  class_weights = calculate_weights(list_labels)

  src = (ImageList.from_folder(path_unlabeled)
          .filter_by_func(lambda x: x not in train_set)
          .split_by_folder(valid="test"))

  src.train._label_list = MultiTransformLabelList

  print("Data unlabeled loading ...")
  print_and_log("Data unlabeled loading ...")

  data_unlabeled = (src.label_from_folder()
          .transform(get_transforms(do_flip = True, flip_vert = True, max_zoom=1, max_warp=None, p_affine=0, p_lighting = 0),
                               size=IMAGE_SIZE)
          .databunch(bs=BATCH_SIZE,collate_fn=MixmatchCollate, num_workers=WORKERS)
          .normalize((meanDatasetComplete_unlabeled, stdDatasetComplete_unlabeled)))
  
  print("Information for unlabeled training data: ")
  print_and_log("Information for unlabeled training data: ")
  list_labels_unlabeled = data_unlabeled.train_ds.y.items
  calculate_weights(list_labels_unlabeled)
  
  #Databunch with all images labeled, for baseline

  print("Data full loading ...")
  print_and_log("Data full loading ...")

  data_full = (ImageList.from_folder(path_labeled)
          .split_by_folder(valid="test")
          .label_from_folder()
          .transform(get_transforms(do_flip = True, flip_vert = True, max_zoom=1, max_warp=None, p_affine=0, p_lighting = 0),
                               size=IMAGE_SIZE)
          .databunch(bs=BATCH_SIZE, num_workers=WORKERS)
          .normalize((meanDatasetComplete_unlabeled, stdDatasetComplete_unlabeled)))

  
  return (data_labeled, data_unlabeled, data_full)

def train_model(model_name, mode, dataset_name, path_labeled, path_unlabeled="", desired_labeled_classes_dist=[.5,.5]):
    
    # Define loss and metrics
    metrics=[accuracy, BalancedAccuracy(), GMean(), HMean(), MatthewsCorreff(), error_rate, FBeta(average='binary', eps=1e-09,beta=2), Precision(average='binary',eps=1e-09), Recall(average='binary',eps=1e-09), Specificity(), AUROC()]
    (data_labeled, data_unlabeled, data_full)= load_data(dataset_name, mode, path_labeled, path_unlabeled, desired_labeled_classes_dist=desired_labeled_classes_dist)
    loss = nn.CrossEntropyLoss(weight=class_weights.float()) 
    

    # Select model
    if (model_name == "densenet_pre"):
        model = models.densenet121( pretrained=True, drop_rate=0.2)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, NUMBER_CLASSES)
    elif (model_name == "densenet"):
        model = models.densenet121(num_classes=NUMBER_CLASSES)
    elif (model_name == "resnet_pre"):
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUMBER_CLASSES)
    elif (model_name == "resnet"):
        model = models.resnet152(num_classes=NUMBER_CLASSES)     
    
        
    model_file_name = ""
    # Select mode
    if (mode == "fully_supervised"):
        print("TRAINING FULLY SUPERVISED MODEL")
        print_and_log("TRAINING FULLY SUPERVISED MODEL")
        model_file_name = "FULLY_SUPERVISED_model_batch_"
        learn = Learner(data_full, model, metrics=metrics, callback_fns = [CSVLogger], path=DIR_MODELS, model_dir=RUN_NAME)

    elif (mode == "partial_supervised"):
        print("TRAINING PARTIALLY SUPERVISED MODEL")
        print_and_log("TRAINING PARTIALLY SUPERVISED MODEL")
        model_file_name = "NO_SSDL_model_batch_"
        if (BALANCED == 5):
          learn = Learner(data_labeled, model, metrics=metrics, loss_func=loss, callback_fns=[CSVLogger], path=DIR_MODELS, model_dir=RUN_NAME)
        else:
          learn = Learner(data_labeled, model, metrics=metrics, callback_fns=[CSVLogger], path=DIR_MODELS, model_dir=RUN_NAME)
        
    elif (mode == "semi_supervised"):
        print("TRAINING SSDL MODEL")
        print_and_log("TRAINING SSDL MODEL")
        model_file_name = "SSDL_model_batch_"
        mixloss = MixupLoss()
        setattr(mixloss, 'reduction', 'none')
        learn = Learner(data_unlabeled, model, loss_func=mixloss, callback_fns=[MixMatchTrainer, CSVLogger], metrics=metrics, path=DIR_MODELS, model_dir=RUN_NAME)
    
    

    model_file_name = model_file_name + path_labeled.split('_')[-1] + "_" + str(NUMBER_LABELED_OBSERVATIONS)


    now = datetime.now()
    print("Starting training at " + now.strftime("%d/%m/%Y %H:%M:%S"))
    print_and_log("Starting training at " + now.strftime("%d/%m/%Y %H:%M:%S"))

    #learn.fit_one_cycle(EPOCHS, LEARNING_RATE, wd=WEIGHT_DECAY,
                        #callbacks=[CSVLogger(learn), SaveModelCallback(learn, monitor='g_mean'), EarlyStoppingCallback(learn, monitor='g_mean', min_delta=0.01, patience=30)])
    learn.fit_one_cycle(EPOCHS, LEARNING_RATE, wd=WEIGHT_DECAY,
                        callbacks=[CSVLogger(learn)])



    now = datetime.now()
    print("Ending training at " + now.strftime("%d/%m/%Y %H:%M:%S"))
    print_and_log("Ending training at " + now.strftime("%d/%m/%Y %H:%M:%S"))
    
    print("Calculating Classification Metrics CSV")
    print_and_log("Calculating Classification Metrics CSV")
    calculate_metrics(learn, mode, path_labeled, dataset_name, NUMBER_LABELED_OBSERVATIONS) #OJOOO
    #File "/home/sacalderon/Johan/mixmatch_new.py", line 993, in train_model
    #calculate_metrics(learn, mode, path_labeled, dataset_name, NUMBER_LABELED_OBSERVATIONS)
    #TypeError: calculate_metrics() missing 1 required positional argument: 'model_id'
    
    
    ##################################################################################################################3
    # read and visualize the image

    
    #image1 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/0/vlcsnap-2021-05-24-10h10m00s686.png')
    #image2 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/0/vlcsnap-2020-10-22-14h34m16s347.png')
    #image3 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/0/vlcsnap-2021-05-24-10h21m39s517.png')
    #image4 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/0/vlcsnap-2021-05-24-10h13m04s218.png')
    image9 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/0/vlcsnap-2020-11-19-22h57m49s793.png')
    image10 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/0/vlcsnap-2021-05-24-09h55m09s506.png')
    image11 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/0/vlcsnap-2021-05-24-09h55m14s801.png')
    image12 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/0/vlcsnap-2021-05-24-09h55m51s427.png')
    
    #image5 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/1/vlcsnap-2020-11-19-21h35m11s308.png')
    #image6 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/1/vlcsnap-2021-05-24-10h12m12s710.png')
    #image7 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/1/vlcsnap-2020-10-22-13h15m54s325.png')
    #image8 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/1/vlcsnap-2020-11-19-22h20m30s795.png')
    image13 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/1/vlcsnap-2020-10-26-10h43m14s590.png')
    image14 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/1/vlcsnap-2020-11-16-10h28m12s621.png')
    image15 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/1/vlcsnap-2020-11-19-14h57m30s693.png')
    image16 = cv2.imread('/home/sacalderon/Johan/Hen_paper/labeled/batch_0/test/1/vlcsnap-2020-11-19-21h36m10s106.png')
    
    images = [image9, image10, image11, image12, image13, image14, image15, image16]
    
    cont = 9
    
    for image in images:
    
    	orig_image = image.copy()
    	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    	height, width, _ = image.shape
    	modelo = learn.model
    	modelo.eval()
    	features_blobs = []
    	def hook_feature(module, input, output):
        	features_blobs.append(output.data.cpu().numpy())
    	modelo._modules.get('features').register_forward_hook(hook_feature)
# get the softmax weight
    	params = list(modelo.parameters())
    	weight_softmax = np.squeeze(params[-2].cpu().data.numpy())     #change

# define the transforms, resize => tensor => normalize
    	transforms1 = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.4040, 0.3708, 0.3128],std=[0.1787, 0.1832, 0.1813])])

# apply the image transforms
    	image_tensor = transforms1(image)
# add batch dimension
    	image_tensor = image_tensor.unsqueeze(0)
    
    	image_tensor = image_tensor.cuda()
    
# forward pass through model
    	outputs = modelo(image_tensor)
# get the softmax probabilities
    	probs = F.softmax(outputs).data.squeeze()
# get the class indices of top k probabilities
#class_idx = topk(probs, 1)[1].int()

# generate class activation mapping for the top1 prediction
    	CAMs = returnCAM(features_blobs[0], weight_softmax, 1)
# file name to save the resulting CAM image with
#save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
#save_name = f"{prueba_heatmap}"
# show and save the results
    	show_cam(CAMs, width, height, orig_image, 1, 1,cont)
    	
    	cont = cont + 1
#######################################################################################################
        
    if (args.save_weights):
        # Store the models' best iteration
        path_weights = DIR_RUN_MODELS + "/" + model_file_name
        print("Saving weights in: " + str(path_weights))
        print_and_log("Saving weights in: " + str(path_weights))
        learn.save(path_weights, with_opt=False)

     	
"""#Execution"""

# Hyperparameters
K_VALUE = args.K_transforms                       # augmentation value 2
T_VALUE = args.T_sharpening                       # sharpen value
ALPHA_VALUE = args.alpha_mix                      # alpha value
LAMBDA_VALUE = args.lambda_unsupervised           # lambda value for MixUp
BALANCED = args.balanced

rampup_coefficient = args.rampup_coefficient 

EPOCHS = args.epochs
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay

NUMBER_LABELED_OBSERVATIONS = args.number_labeled
IMAGE_SIZE = args.size_image
BATCH_SIZE = args.batch_size
WORKERS = args.workers

NUMBER_CLASSES = args.num_classes

desired_labeled_classes_dist = [float(s) for s in args.desired_labeled_classes_dist.split(',')]


RUN_NAME = args.RUN_NAME

DIR_MODELS = args.DIR_MODELS
DIR_RUN_MODELS = DIR_MODELS + "/" + RUN_NAME

DIR_SUMMARIES = args.DIR_SUMMARIES
DIR_RUN_SUMMARIES = DIR_SUMMARIES + RUN_NAME

if not os.path.exists(DIR_RUN_MODELS):
        os.mkdir(DIR_RUN_MODELS)
if not os.path.exists(DIR_RUN_SUMMARIES):
        os.mkdir(DIR_RUN_SUMMARIES)
        
print("Parameters: ", args)
print_and_log(str(args))



train_model(args.model, args.mode, args.norm_stats, args.path_labeled, args.path_unlabeled, desired_labeled_classes_dist=desired_labeled_classes_dist )


