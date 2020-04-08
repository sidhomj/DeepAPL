import os
import glob
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB, IMREAD_COLOR, COLOR_BGR2GRAY,copyMakeBorder
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import colorsys
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score,accuracy_score
import scipy.stats
from DeepAPL.functions.color_norm import *

def Get_Images(sub_dir,sample,include_cell_types=None,exclude_cell_types=None,load_images=True,color_norm=True):

    sub_dir_2 = os.path.join(sub_dir,'Signed slides')
    type_list = os.listdir(sub_dir_2)
    if include_cell_types is not None:
        type_list = include_cell_types
    if exclude_cell_types is not None:
        type_list = list(set(type_list) - set(exclude_cell_types))

    cell_type = []
    files = []
    imgs = []

    for type in type_list:
        files_read = glob.glob(os.path.join(sub_dir_2, type, '*jpg'))
        for file in files_read:
            files.append(file)
            cell_type.append(type)
            if load_images is True:
                img = imread(file)
                img = resize(cvtColor(img, COLOR_BGR2RGB),(360,360))
                #img = resize(cvtColor(img, COLOR_BGR2GRAY),(360,360))
                #img = np.expand_dims(img,-1)
                #img = img / 255
                img = img.astype('float32')
                # img = np.expand_dims(img, 0)
                imgs.append(img)


    files = np.asarray(files)
    cell_type = np.asarray(cell_type)
    cell_type_raw = cell_type

    if len(imgs) != 0:
        if color_norm:
            # normalize imgs
            cns = ColorNormStains()
            cns.process_img_data(imgs)
            imgs = cns.Get_Normed_Data()
        imgs = np.stack(imgs,axis=0)

        if sample is not None:
            if len(imgs) > sample:
                idx = list(range(len(imgs)))
                idx = np.random.choice(idx, sample, replace=False)
                imgs = imgs[idx]
                files = files[idx]
                cell_type = cell_type[idx]

        imgs = np.expand_dims(imgs,0)

    return imgs, files, cell_type, cell_type_raw

def Get_Train_Valid_Test(Vars,Y=None,test_size=0.25,regression=False,LOO = None):

    if regression is False:
        var_train = []
        var_valid = []
        var_test = []
        if Y is not None:
            y_label = np.argmax(Y,1)
            classes = list(set(y_label))

            if LOO is None:
                for ii, type in enumerate(classes, 0):
                    idx = np.where(y_label == type)[0]
                    if idx.shape[0] == 0:
                        continue

                    np.random.shuffle(idx)
                    train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
                    idx = np.setdiff1d(idx, train_idx)
                    np.random.shuffle(idx)
                    half_val_len = int(idx.shape[0] * 0.5)
                    valid_idx, test_idx = idx[:half_val_len], idx[half_val_len:]

                    if ii == 0:
                        for var in Vars:
                            var_train.append(var[train_idx])
                            var_valid.append(var[valid_idx])
                            var_test.append(var[test_idx])

                        var_train.append(Y[train_idx])
                        var_valid.append(Y[valid_idx])
                        var_test.append(Y[test_idx])
                    else:
                        for jj, var in enumerate(Vars, 0):
                            var_train[jj] = np.concatenate((var_train[jj], var[train_idx]), 0)
                            var_valid[jj] = np.concatenate((var_valid[jj], var[valid_idx]), 0)
                            var_test[jj] = np.concatenate((var_test[jj], var[test_idx]), 0)

                        var_train[-1] = np.concatenate((var_train[-1], Y[train_idx]), 0)
                        var_valid[-1] = np.concatenate((var_valid[-1], Y[valid_idx]), 0)
                        var_test[-1] = np.concatenate((var_test[-1], Y[test_idx]), 0)

            else:
                idx = list(range(len(Y)))
                if LOO ==1:
                    test_idx = np.random.choice(idx, LOO, replace=False)[0]
                else:
                    test_idx = np.random.choice(idx, LOO, replace=False)

                train_idx = np.setdiff1d(idx,test_idx)
                valid_idx = test_idx

                for var in Vars:
                    var_train.append(var[train_idx])
                    if LOO ==1:
                        var_valid.append(np.expand_dims(var[valid_idx], 0))
                        var_test.append(np.expand_dims(var[test_idx], 0))
                    else:
                        var_valid.append(var[valid_idx])
                        var_test.append(var[test_idx])


                var_train.append(Y[train_idx])

                if LOO == 1:
                    var_valid.append(np.expand_dims(Y[valid_idx], 0))
                    var_test.append(np.expand_dims(Y[test_idx], 0))
                else:
                    var_valid.append(Y[valid_idx])
                    var_test.append(Y[test_idx])


    else:
        idx = np.asarray(list(range(len(Y))))
        np.random.shuffle(idx)
        train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
        idx = np.setdiff1d(idx, train_idx)
        np.random.shuffle(idx)
        half_val_len = int(idx.shape[0] * 0.5)
        valid_idx, test_idx = idx[:half_val_len], idx[half_val_len:]

        var_train = []
        var_valid = []
        var_test = []
        for var in Vars:
            var_train.append(var[train_idx])
            var_valid.append(var[valid_idx])
            var_test.append(var[test_idx])

        var_train.append(Y[train_idx])
        var_valid.append(Y[valid_idx])
        var_test.append(Y[test_idx])



    return var_train,var_valid,var_test

def Get_Train_Test(Vars,test_idx,train_idx,Y=None):
    var_train = []
    var_test = []
    for var in Vars:
        var_train.append(var[train_idx])
        var_test.append(var[test_idx])

    if Y is not None:
        var_train.append(Y[train_idx])
        var_test.append(Y[test_idx])

    return var_train, var_test

def get_batches(Vars, batch_size=10,random=False):
    """ Return a generator that yields batches from vars. """
    #batch_size = len(x) // n_batches
    x = Vars[0]
    if len(x) % batch_size == 0:
        n_batches = (len(x) // batch_size)
    else:
        n_batches = (len(x) // batch_size) + 1

    sel = np.asarray(list(range(x.shape[0])))
    if random is True:
        np.random.shuffle(sel)

    for ii in range(0, n_batches * batch_size, batch_size):
        # If we're not on the last batch, grab data with size batch_size
        if ii != (n_batches - 1) * batch_size:
            sel_ind=sel[ii: ii + batch_size]
        else:
            sel_ind = sel[ii:]

        Vars_Out = [var[sel_ind] for var in Vars]

        yield Vars_Out

def Generate_Colors(N):
    HSV_tuples = [(x * 1.0 / N, 1.0, 0.5) for x in range(N)]
    np.random.shuffle(HSV_tuples)
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples

def imscatter(x, y, image, ax=None, zoom=1,bordersize=None,border_color=None):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    if bordersize is not None:
        image = copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                borderType=cv2.BORDER_CONSTANT, value=np.round(255*border_color))

    im = OffsetImage(image, zoom=zoom)

    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def stop_check(loss,stop_criterion,stop_criterion_window):
    w = loss[-stop_criterion_window:]
    return (w[0]-w[-1])/w[0] < stop_criterion

def KNN(distances,labels,files,patients,k=1,folds=5,metrics=['Recall','Precision','F1_Score','AUC'],n_jobs=1):
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)

    if folds > np.min(np.bincount(labels)):
        skf = KFold(n_splits=folds, random_state=None, shuffle=True)
    else:
        skf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=True)

    neigh = KNeighborsClassifier(n_neighbors=k, metric='precomputed', weights='distance',n_jobs=n_jobs)

    pred_list = []
    pred_prob_list = []
    labels_list = []
    files_list = []
    patients_list = []

    patients_ref = np.unique(patients)
    labels_ref = np.asarray([labels[np.where(patients==x)[0][0]] for x in patients_ref])

    for train_idx,test_idx in skf.split(patients_ref,labels_ref):
        train_idx = np.where(np.isin(patients,patients_ref[train_idx]))[0]
        test_idx = np.where(np.isin(patients,patients_ref[test_idx]))[0]
        #for train_idx, test_idx in skf.split(distances,labels):
        distances_train = distances[train_idx, :]
        distances_train = distances_train[:, train_idx]

        distances_test = distances[test_idx, :]
        distances_test = distances_test[:, train_idx]

        labels_train = labels[train_idx]
        labels_test = labels[test_idx]

        neigh.fit(distances_train, labels_train)
        pred = neigh.predict(distances_test)
        pred_prob = neigh.predict_proba(distances_test)

        labels_list.extend(labels_test)
        pred_list.extend(pred)
        pred_prob_list.extend(pred_prob)
        files_list.extend(files[test_idx])
        patients_list.extend(patients[test_idx])

    pred = np.asarray(pred_list)
    pred_prob = np.asarray(pred_prob_list)
    labels = np.asarray(labels_list)
    labels_out = labels
    files = np.asarray(files_list)
    patients = np.asarray(patients_list)

    OH = OneHotEncoder(sparse=False,categories='auto')
    labels = OH.fit_transform(labels.reshape(-1,1))
    pred = OH.transform(pred.reshape(-1,1))

    metric = []
    value = []
    classes=[]
    k_list = []
    for ii,c in enumerate(lb.classes_):
        if 'Recall' in metrics:
            value.append(recall_score(y_true=labels[:,ii],y_pred=pred[:,ii]))
            metric.append('Recall')
            classes.append(c)
            k_list.append(k)
        if 'Precision' in metrics:
            value.append(precision_score(y_true=labels[:,ii],y_pred=pred[:,ii]))
            metric.append('Precision')
            classes.append(c)
            k_list.append(k)
        if 'F1_Score' in metrics:
            value.append(f1_score(y_true=labels[:, ii], y_pred=pred[:,ii]))
            metric.append('F1_Score')
            classes.append(c)
            k_list.append(k)
        if 'AUC' in metrics:
            value.append(roc_auc_score(labels[:, ii],pred_prob[:,ii]))
            metric.append('AUC')
            classes.append(c)
            k_list.append(k)


    return classes,metric,value,k_list,files,labels_out,patients,pred_prob

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h,h