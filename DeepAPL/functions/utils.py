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
import copy
from skimage import io
from skimage.morphology import binary_closing, disk, binary_dilation, remove_small_objects
import scipy.ndimage as nd
import tensorflow as tf

def Get_Images(sub_dir,sample,include_cell_types=None,exclude_cell_types=None,load_images=True,color_norm=True,nmf=False,rm_rbc=False):

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
                #img = np.expand_dims(img,-1)
                img = img.astype('float32')
                # img = np.expand_dims(img, 0)
                imgs.append(img)


    files = np.asarray(files)
    cell_type = np.asarray(cell_type)
    cell_type_raw = cell_type

    if sample is not None:
        if len(imgs) > sample:
            idx = list(range(len(imgs)))
            idx = np.random.choice(idx, sample, replace=False)
            imgs = list(np.array(imgs)[idx])
            files = files[idx]
            cell_type = cell_type[idx]

    if len(imgs) != 0:
        if color_norm:
            # normalize imgs
            cns = ColorNormStains()
            cns.process_img_data(imgs)
            imgs_norm,imgs_nmf = cns.Get_Normed_Data()
            if rm_rbc:
                #get masks
                masks = [create_mask_rbc(x) for x in imgs_nmf]

                imgs = np.stack(imgs_norm, axis=0)
                masks = np.stack(masks,axis=0)
                imgs[masks] = 1.0
            else:
                imgs = np.stack(imgs_norm, axis=0)

            fig,ax = plt.subplots(10,10,figsize=(10,10))
            ax = np.ndarray.flatten(ax)
            sel = range(len(imgs))
            for a,s in zip(ax,sel):
                a.imshow(imgs[s])
                a.set_xticks([])
                a.set_yticks([])
            plt.tight_layout()

        else:
            imgs = np.stack(imgs, axis=0)
            imgs = imgs / 255

            if rm_rbc:
                cns = ColorNormStains()
                cns.process_img_data(imgs)
                imgs_temp, imgs_nmf = cns.Get_Normed_Data()
                # get masks
                masks = [create_mask_rbc(x) for x in imgs_nmf]
                masks = np.stack(masks, axis=0)
                imgs[masks,:] = cns.I_0

        imgs = np.expand_dims(imgs,0)

    return imgs, files, cell_type, cell_type_raw

def create_mask_rbc_dep(img):
    #get mask for rbc
    mask = copy.copy(img[:,:,2])
    th_c = np.percentile(mask,70)
    mask[mask > th_c] = 1.0
    mask[mask <= th_c] = 0.0
    mask = mask.astype('bool')
    mask = remove_small_objects(mask,min_size=150)
    strel = disk(10)
    mask = binary_closing(mask,strel)
    mask = nd.morphology.binary_fill_holes(mask)
    strel = disk(20)
    mask = binary_dilation(mask, strel)
    mask_rbc = mask

    #get masks for nucleus
    mask = copy.copy(img[:,:,0])
    th_c = np.percentile(mask,95)
    mask[mask > th_c] = 1.0
    mask[mask <= th_c] = 0.0
    mask = mask.astype('bool')
    mask = remove_small_objects(mask,min_size=150)
    strel = disk(10)
    mask = binary_closing(mask,strel)
    mask = nd.morphology.binary_fill_holes(mask)
    strel = disk(10)
    mask = binary_dilation(mask, strel)
    mask_nuc_1 = mask

    mask = copy.copy(img[:,:,1])
    th_c = np.percentile(mask,95)
    mask[mask > th_c] = 1.0
    mask[mask <= th_c] = 0.0
    mask = mask.astype('bool')
    mask = remove_small_objects(mask,min_size=150)
    strel = disk(10)
    mask = binary_closing(mask,strel)
    mask = nd.morphology.binary_fill_holes(mask)
    strel = disk(10)
    mask = binary_dilation(mask, strel)
    mask_nuc_2 = mask
    mask_nuc = (mask_nuc_1.astype(int) + mask_nuc_2.astype(int))
    mask_nuc[mask_nuc>1] = 1

    mask = mask_rbc.astype(int) - mask_nuc
    mask[mask<0] = 0

    fig,ax = plt.subplots(2,3,figsize=(10,5))
    ax = np.ndarray.flatten(ax)
    ax[0].imshow(img[:,:,0])
    ax[0].set_title('channel 1')
    ax[1].imshow(img[:,:,1])
    ax[1].set_title('channel 2')
    ax[2].imshow(img[:,:,2])
    ax[2].set_title('channel 3')
    ax[3].imshow(img[:,:,0]*img[:,:,1])
    ax[3].set_title('1 * 2')
    ax[4].imshow(img[:,:,1]*img[:,:,2])
    ax[4].set_title('2 * 3')
    ax[5].imshow(img[:,:,0]*img[:,:,2])
    ax[5].set_title('1 * 3')
    plt.tight_layout()

    plt.figure()
    plt.imshow(img[:,:,0])
    plt.figure()
    plt.imshow(img[:,:,1])
    plt.figure()
    plt.imshow(img[:,:,0]*img[:,:,1])
    plt.figure()




    return mask.astype('bool')

def create_mask_rbc(img):
    #get mask for rbc
    mask = copy.copy(img[:,:,2])
    th_c = np.percentile(mask,70)
    th_c = 0.008
    mask[mask > th_c] = 1.0
    mask[mask <= th_c] = 0.0
    mask = mask.astype('bool')
    mask = remove_small_objects(mask,min_size=150)
    strel = disk(10)
    mask = binary_closing(mask,strel)
    mask = nd.morphology.binary_fill_holes(mask)
    strel = disk(20)
    mask = binary_dilation(mask, strel)
    mask_rbc = mask

    #get masks for nucleus
    mask = copy.copy(img[:,:,0])*copy.copy(img[:,:,1])
    th_c = np.percentile(mask,95)
    th_c = 0.006
    mask[mask > th_c] = 1.0
    mask[mask <= th_c] = 0.0
    mask = mask.astype('bool')
    mask = remove_small_objects(mask,min_size=150)
    strel = disk(10)
    mask = binary_closing(mask,strel)
    mask = nd.morphology.binary_fill_holes(mask)
    strel = disk(10)
    mask = binary_dilation(mask, strel)
    mask_nuc = mask

    mask = mask_rbc.astype(int) - mask_nuc.astype(int)
    mask[mask<0] = 0
    mask = mask.astype('bool')
    return mask

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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

def Run_Graph_WF(set,sess,self,GO,batch_size,random=True,
                 train=True,drop_out_rate=None,multisample_dropout_rate=None,
                 class_weights = None,subsample=None):
    loss = []
    accuracy = []
    predicted_list = []
    for vars in get_batches(set, batch_size=batch_size, random=random):
        if subsample is None:
            var_idx = np.where(np.isin(self.patients, vars[0]))[0]
        else:
            var_idx = []
            if subsample is not None:
                for p in np.unique(vars[0]):
                    vidx = np.where(np.isin(self.patients,p))[0]
                    if len(vidx)>subsample:
                        vidx = np.random.choice(vidx,subsample,replace=False)
                    var_idx.append(vidx)
            var_idx = np.hstack(var_idx)

        lb = LabelEncoder()
        lb.fit(vars[0])
        _,_,sample_idx = np.intersect1d(lb.classes_,vars[0],return_indices=True)
        vars = [v[sample_idx] for v in vars]
        i = lb.transform(self.patients[var_idx])

        OH = OneHotEncoder(categories='auto')
        sp = OH.fit_transform(i.reshape(-1, 1)).T
        sp = sp.tocoo()
        indices = np.mat([sp.row, sp.col]).T
        sp = tf.SparseTensorValue(indices, sp.data, sp.shape)

        feed_dict = {GO.Y: vars[-1],
                     GO.sp: sp,
                     GO.class_weights: class_weights}

        feed_dict[GO.X] = self.imgs[var_idx]

        if drop_out_rate is not None:
            feed_dict[GO.prob] = drop_out_rate

        if multisample_dropout_rate is not None:
            feed_dict[GO.prob_multisample] = multisample_dropout_rate

        if train:
            loss_i, accuracy_i, _, predicted_i = sess.run([GO.loss, GO.accuracy, GO.opt, GO.predicted],
                                                          feed_dict=feed_dict)
        else:
            loss_i, accuracy_i, predicted_i = sess.run([GO.loss, GO.accuracy, GO.predicted],
                                                       feed_dict=feed_dict)

        loss.append(loss_i)
        accuracy.append(accuracy_i)
        pred_temp = np.zeros_like(predicted_i)
        pred_temp[sample_idx] = predicted_i
        predicted_i = pred_temp
        predicted_list.append(predicted_i)

    loss = np.mean(loss)
    accuracy = np.mean(accuracy)
    predicted_out = np.vstack(predicted_list)
    try:
        auc = roc_auc_score(set[-1], predicted_out)
    except:
        auc = 0.0
    return loss,accuracy,predicted_out,auc

def Get_Cell_Pred(self,batch_size,GO,sess):
    predicted_list = []
    i = np.asarray(range(len(self.Y)))
    idx = []
    for vars in get_batches(self.test, batch_size=batch_size, random=False):
        var_idx = np.where(np.isin(self.patients, vars[0]))[0]
        OH = OneHotEncoder(categories='auto')
        sp = OH.fit_transform(i[var_idx].reshape(-1, 1)).T
        sp = sp.tocoo()
        indices = np.mat([sp.row, sp.col]).T
        sp = tf.SparseTensorValue(indices, sp.data, sp.shape)

        feed_dict = {GO.sp: sp,
                     GO.X: self.imgs[var_idx]}

        predicted_list.append(sess.run(GO.predicted,feed_dict=feed_dict))
        idx.append(var_idx)

    return np.vstack(predicted_list),np.squeeze(np.hstack(idx))
