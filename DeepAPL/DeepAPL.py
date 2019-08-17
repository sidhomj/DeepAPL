import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import os
import glob
import pickle
import matplotlib.pyplot as plt
from DeepAPL.functions.utils import *
from DeepAPL.functions.Layers import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import umap
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import warnings
from scipy.spatial.distance import squareform, pdist
from copy import deepcopy
import scipy

class base(object):
    def __init__(self,Name='tr_obj',device='/device:GPU:0'):
        self.Name = Name
        self.device = device

        #Create directory for results of analysis
        directory = self.Name + '_Results'
        self.directory_results = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        #Create directory for any temporary files
        directory = self.Name
        if not os.path.exists(directory):
            os.makedirs(directory)

        #Create directory for models
        if not os.path.exists(os.path.join(self.Name,'models')):
            os.makedirs(os.path.join(self.Name,'models'))

    def Import_Data(self,directory,Load_Prev_Data=False,classes=None,sample=None,
                    include_cell_types=None,exclude_cell_types=None,save_data=True):
        if Load_Prev_Data is False:

            if classes is None:
                classes = [d for d in os.listdir(directory) if os.path.isdir(directory + d)]
                classes = [f for f in classes if not f.startswith('.')]

            self.lb = LabelEncoder()
            self.lb.fit(classes)
            self.classes = self.lb.classes_

            labels = []
            imgs = []
            patients = []
            cell_type = []
            files = []
            check=0

            for type in self.classes:
                pts = os.listdir(os.path.join(directory,type))
                for pt in pts:
                    sub_dir = os.path.join(directory,type,pt)
                    list_dir = sorted(os.listdir(sub_dir))
                    if 'Signed slides' in list_dir:
                        imgs_temp,files_temp,cell_type_temp,cell_type_raw_temp = Get_Images(sub_dir,sample,include_cell_types,exclude_cell_types)
                    else:
                        sub_dir_2 = os.path.join(sub_dir,list_dir[0])
                        imgs_temp,files_temp,cell_type_temp,cell_type_raw_temp = Get_Images(sub_dir_2,sample,include_cell_types,exclude_cell_types)

                    if not isinstance(imgs_temp,list):
                        imgs.append(np.squeeze(imgs_temp,0))
                        labels.append([type]*imgs_temp.shape[1])
                        patients.append([pt]*imgs_temp.shape[1])
                        cell_type.append(cell_type_temp)
                        files.append(files_temp)
                    else:
                        check +=1
                        if check==2:
                            check=2


            imgs = np.vstack(imgs)
            labels = np.hstack(labels)
            patients = np.hstack(patients)
            cell_type = np.hstack(cell_type)
            files = np.hstack(files)

            Y = self.lb.transform(labels)
            OH = OneHotEncoder(sparse=False)
            Y = OH.fit_transform(Y.reshape(-1,1))

            if save_data is True:
                np.save(os.path.join(self.Name, 'imgs'), imgs)
                with open(os.path.join(self.Name, 'data.pkl'), 'wb') as f:
                    pickle.dump([Y,labels,patients,cell_type,files,self.lb], f, protocol=4)


        else:

            imgs = np.load(os.path.join(self.Name, 'imgs.npy'))
            with open(os.path.join(self.Name,'data.pkl'),'rb') as f:
                Y,labels,patients,cell_type,files,self.lb = pickle.load(f)


        self.imgs = imgs
        self.labels = labels
        self.patients = patients
        self.cell_type = cell_type
        self.files = files
        self.Y = Y
        self.predicted = np.zeros((len(self.Y),len(self.lb.classes_)))

    def AUC_Curve(self):
        class_lb = self.lb.classes_.tolist()
        plt.figure()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC_Curves')

        ROC_DFs = []
        for ii in class_lb:
            id = ii
            id_num = class_lb.index(id)

            roc_score = roc_auc_score(self.y_test[:, id_num], self.y_pred[:, id_num])
            fpr, tpr, th = roc_curve(self.y_test[:, id_num], self.y_pred[:, id_num])
            roc_df = pd.DataFrame()
            roc_df['th'] = th
            roc_df['fpr'] = fpr
            roc_df['rpr'] = tpr
            ROC_DFs.append(roc_df)
            plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (id, roc_score))

        plt.legend(loc="lower right")
        plt.show()
        self.ROC = dict(zip(class_lb,ROC_DFs))

class DeepAPL_SC(base):
    def Get_Train_Valid_Test(self,test_size=0.25,combine_train_valid=False,train_all=False):
        patients = np.unique(self.patients)
        Y = np.asarray([self.Y[np.where(self.patients == x)[0][0]] for x in patients])
        train,valid,test = Get_Train_Valid_Test([patients],Y,test_size=test_size)

        self.train_idx =np.where(np.isin(self.patients,train[0]))[0]
        self.valid_idx = np.where(np.isin(self.patients,valid[0]))[0]
        self.test_idx = np.where(np.isin(self.patients,test[0]))[0]

        self.train = []
        self.valid = []
        self.test = []

        Vars = [self.imgs,self.patients,self.files,self.Y]

        for v in Vars:
            self.train.append(v[self.train_idx])
            self.valid.append(v[self.valid_idx])
            self.test.append(v[self.test_idx])

        if combine_train_valid:
            train = []
            for i in range(len(self.train)):
                train.append(np.concatenate((self.train[i],self.valid[i]),0))
            self.train = train
            self.valid = self.test

        if train_all:
            train = []
            for i in range(len(self.train)):
                train.append(np.concatenate((self.train[i],self.valid[i],self.test[i]),0))
            self.train = train

    def Train(self,batch_size = 10, epochs_min = 10,
              stop_criterion=0.001,stop_criterion_window=10,
              num_fc_layers=0,num_units=256,weight_by_class=False,drop_out_rate=0.0,iteration=0):

        GO = graph_object()
        with tf.device(self.device):
            graph_model = tf.Graph()
            with graph_model.as_default():
                GO.Y = tf.placeholder(dtype=tf.float32,shape=[None,self.Y.shape[1]])
                Conv_Model(GO, self)
                attention=True
                if attention:
                    logits = Attention_Layer(GO)
                else:
                    # fc = tf.concat((tf.reduce_max(GO.l1,axis=[1,2]),
                    #            tf.reduce_max(GO.l2, axis=[1, 2]),
                    #            tf.reduce_max(GO.l3, axis=[1, 2]),
                    #            tf.red#uce_max(GO.l4, axis=[1, 2])),axis=1)
                    fc = tf.reduce_mean(GO.l3,axis=[1,2])
                    # WM = tf.exp(tf.layers.dense(GO.l4,GO.l4.shape[-1],activation=tf.nn.sigmoid))
                    # WM = WM/tf.reduce_sum(WM,axis=[1,2])[:,tf.newaxis,tf.newaxis,:]
                    # fc = tf.reduce_sum(GO.l4*WM,axis=[1,2])

                    for i in range(num_fc_layers):
                        fc = tf.layers.dense(fc, num_units)
                        fc = tf.layers.dropout(fc, GO.prob)

                    logits = tf.layers.dense(fc,self.Y.shape[1])
                    w = tf.trainable_variables()[-2]

                if weight_by_class:
                    class_weights = tf.constant([(1 / (np.sum(self.Y, 0) / np.sum(self.Y))).tolist()])
                    weights = tf.squeeze(tf.matmul(tf.cast(GO.Y, dtype='float32'), class_weights, transpose_b=True), axis=1)
                else:
                    weights = 1

                loss = tf.reduce_mean(weights*tf.nn.softmax_cross_entropy_with_logits_v2(GO.Y,logits))

                opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

                with tf.name_scope('Accuracy_Measurements'):
                    predicted = tf.nn.softmax(logits, name='predicted')
                    correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(GO.Y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

                saver = tf.train.Saver()

                #self.var_names = [GO.w.name,GO.l1.name,GO.l2.name,GO.l3.name]

        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        training = True
        e=1
        stop_check_list = []
        val_loss_total = []
        with tf.Session(graph=graph_model, config=config) as sess:
            sess.run(tf.global_variables_initializer())
            while training is True:
                Vars = [self.train[0],self.train[3]]
                loss_temp = []
                for vars in get_batches(Vars, batch_size=batch_size, random=True):
                    feed_dict = {GO.X: vars[0],
                                 GO.Y: vars[1],
                                 GO.prob:drop_out_rate}
                    loss_i,_  = sess.run([loss,opt],feed_dict=feed_dict)
                    loss_temp.append(loss_i)

                train_loss = np.mean(loss_temp)

                Vars = [self.valid[0],self.valid[3]]
                loss_temp = []
                for vars in get_batches(Vars, batch_size=batch_size, random=False):
                    feed_dict = {GO.X: vars[0],
                                 GO.Y: vars[1]}
                    loss_i  = sess.run(loss,feed_dict=feed_dict)
                    loss_temp.append(loss_i)

                valid_loss = np.mean(loss_temp)
                val_loss_total.append(valid_loss)

                Vars = [self.test[0],self.test[3]]
                loss_temp = []
                pred_temp = []
                for vars in get_batches(Vars, batch_size=batch_size, random=False):
                    feed_dict = {GO.X: vars[0],
                                 GO.Y: vars[1]}
                    loss_i,pred_i  = sess.run([loss,predicted],feed_dict=feed_dict)
                    loss_temp.append(loss_i)
                    pred_temp.append(pred_i)

                test_loss = np.mean(loss_temp)
                pred_temp = np.vstack(pred_temp)
                self.y_pred = pred_temp
                self.y_test = Vars[1]
                test_auc = roc_auc_score(Vars[1],pred_temp)

                print("Training_Statistics: \n",
                      "Epoch: {}".format(e),
                      "Training loss: {:.5f}".format(train_loss),
                      "Validation loss: {:.5f}".format(valid_loss),
                      "Testing loss: {:.5f}".format(test_loss),
                      "Testing AUC: {:.5}".format(test_auc))

                stop_check_list.append(stop_check(val_loss_total, stop_criterion, stop_criterion_window))
                if e > epochs_min:
                    if np.sum(stop_check_list[-3:]) >= 3:
                        break

                e +=1

            Vars = [self.imgs]
            w = []
            l1 = []
            l2 = []
            l3 = []
            for vars in get_batches(Vars, batch_size=batch_size, random=False):
                feed_dict = {GO.X: vars[0]}
                w_temp,l1_temp, l2_temp, l3_temp = sess.run([GO.w,GO.l1,GO.l2,GO.l3],feed_dict=feed_dict)
                w.append(w_temp)
                l1.append(l1_temp)
                l2.append(l2_temp)
                l3.append(l3_temp)

            self.w = np.vstack(w)
            self.l1 = np.vstack(l1)
            self.l2 = np.vstack(l2)
            self.l3 = np.vstack(l3)

            self.predicted[self.test_idx] += self.y_pred
            saver.save(sess, os.path.join(self.Name, 'models','model_'+str(iteration),'model.ckpt'))

    def Get_Cell_Predicted(self,confidence=0.95):
        df = pd.DataFrame()
        df['Files'] = self.files
        df['Label'] = self.labels
        for ii,c in enumerate(self.lb.classes_,0):
            df[c] = self.predicted[:,ii]

        if hasattr(self,'predicted_dist'):
            for ii, c in enumerate(self.lb.classes_, 0):
                # compute CI for predictions
                predicted_dist = self.predicted_dist[:, :, ii]
                ci = []
                for d in predicted_dist.T:
                    ci.append(mean_confidence_interval(d,confidence=confidence))
                ci = np.vstack(ci)
                df[c+'_mean']=ci[:,0]
                df[c+'_low']=ci[:,1]
                df[c+'_high']=ci[:,2]
                df[c+'_ci']=ci[:,3]

        self.Cell_Pred = df

    def Representative_Cells(self,type='APL',num=12):
        df = deepcopy(self.Cell_Pred)
        df.reset_index(inplace=True)
        df.sort_values(by=type,ascending=False,inplace=True)
        df = df[df['Label']==type]
        df.set_index('Files',inplace=True)

        #image_paths = np.asarray(df.index)[0:num]
        idx = np.asarray(df['index'])[0:num]
        prob = np.asarray(df[type])[0:num]
        if hasattr(self, 'predicted_dist'):
            ci = np.asarray(df[type+'_ci'])[0:num]
        else:
            ci=prob
        nrows = int(np.round(np.sqrt(num)))
        ncols = int(num/nrows)

        fig,ax = plt.subplots(nrows=nrows,ncols=ncols)
        ax = np.ndarray.flatten(ax)


        for a,i,p,c in zip(ax,idx,prob,ci):
            a.imshow(self.imgs[i])
            if hasattr(self,'predicted_dist'):
                a.set_title('Prob = '+str(round(p,3))+', CI='+str(round(c,3)))
            else:
                a.set_title('Prob = '+str(round(p,3)))
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlabel('')
            a.set_ylabel('')
        fig.suptitle(type)
        fig.savefig(os.path.join(self.directory_results, type + '_top.png'))

        w = self.w
        t = self.lb.transform([type])[0]
        w = w[:,:,:,t]

        fig,ax = plt.subplots(nrows=nrows,ncols=ncols)
        ax = np.ndarray.flatten(ax)

        for a,i,p,c in zip(ax,idx,prob,ci):
            a.imshow(self.imgs[i])
            a.imshow(w[i], alpha=0.65, cmap='jet')
            if hasattr(self,'predicted_dist'):
                a.set_title('Prob = '+str(round(p,3))+', CI='+str(round(c,3)))
            else:
                a.set_title('Prob = '+str(round(p,3)))
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlabel('')
            a.set_ylabel('')
        fig.suptitle(type)
        fig.savefig(os.path.join(self.directory_results, type + '_top_act.png'))

    def Get_Kernels(self,type='APL',num=12,top_kernels=3,layer=1):
        df = pd.DataFrame()
        df['Files'] = self.files
        df['Label'] = self.labels
        for ii,c in enumerate(self.lb.classes_,0):
            df[c] = self.predicted[:,ii]
        self.Cell_Pred = deepcopy(df)

        df.reset_index(inplace=True)
        df.sort_values(by=type,ascending=False,inplace=True)
        df = df[df['Label']==type]
        df.set_index('Files',inplace=True)

        idx = np.asarray(df['index'])[0:num]

        if layer == 1:
            features = self.l1
        elif layer == 2:
            features = self.l2
        elif layer == 3:
            features = self.l3
        elif layer == 4:
            features = self.l4

        features_norm = MinMaxScaler().fit_transform(np.sum(features, axis=(1, 2)))
        features_sel = features_norm[idx]

        dir = os.path.join(self.directory_results, 'Kernels', type)
        if not os.path.exists(dir):
            os.makedirs(dir)

        file_list = [f for f in os.listdir(dir)]
        [os.remove(os.path.join(dir, f)) for f in file_list]

        for zz, (l, i) in enumerate(zip(features_sel, idx), 0):
            j = np.flip(np.argsort(l))[0:top_kernels]
            k = np.transpose(features[i], axes=[2, 0, 1])
            k = k[j]

            for kk, jj in zip(k, j):
                plt.figure()
                plt.imshow(kk, cmap='jet')
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('')
                plt.ylabel('')
                plt.savefig(os.path.join(dir, str(zz) + '_' + str(jj)))
                plt.close()

    def Monte_Carlo_CrossVal(self,folds=5,test_size=0.25,batch_size = 10, epochs_min = 10,
              stop_criterion=0.001,stop_criterion_window=10,
              num_fc_layers=0,num_units=256,weight_by_class=False,drop_out_rate=0.0,
                             combine_train_valid=False,train_all=False):

        y_pred = []
        y_test = []
        predicted = np.zeros_like(self.predicted)
        counts = np.zeros_like(self.predicted)

        out_dir = os.path.join(self.Name,'ensemble_model')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i in range(0, folds):
            print(i)
            self.Get_Train_Valid_Test(test_size=test_size,combine_train_valid=combine_train_valid,train_all=train_all)
            self.Train(batch_size = batch_size, epochs_min = epochs_min,
              stop_criterion=stop_criterion,stop_criterion_window=stop_criterion_window,
              num_fc_layers=num_fc_layers,num_units=num_units,weight_by_class=weight_by_class,
                       drop_out_rate=drop_out_rate,iteration=i)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            predicted[self.test_idx] += self.y_pred
            counts[self.test_idx] += 1

            y_test2 = np.vstack(y_test)
            y_pred2 = np.vstack(y_pred)

            print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2,1),np.argmax(y_test2,1)))))

            if self.y_test.shape[1] == 2:
                if i > 0:
                    y_test2 = np.vstack(y_test)
                    if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                        print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))


        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        self.predicted = np.divide(predicted,counts, out = np.zeros_like(predicted), where = counts != 0)
        print('Monte Carlo Simulation Completed')

    def Inference(self,model='model_0',batch_size=100):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.import_meta_graph(os.path.join(self.Name, 'models',model, 'model.ckpt.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(self.Name,'models', model)))
            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name('Input:0')
            pred = graph.get_tensor_by_name('Accuracy_Measurements/predicted:0')
            l1_var = graph.get_tensor_by_name('dropout/Identity:0')
            l2_var = graph.get_tensor_by_name('dropout_1/Identity:0')
            l3_var = graph.get_tensor_by_name('dropout_2/Identity:0')
            w_var = graph.get_tensor_by_name('dense/BiasAdd:0')

            predicted = []
            w = []
            l1 = []
            l2 = []
            l3 = []
            for x in get_batches([self.imgs], batch_size=batch_size, random=False):
                feed_dict = {X: x[0]}
                predicted_i,w_temp, l1_temp,l2_temp,l3_temp = sess.run([pred,w_var,l1_var,l2_var,l3_var],feed_dict=feed_dict)
                predicted.append(predicted_i)
                l1.append(l1_temp)
                l2.append(l2_temp)
                l3.append(l3_temp)
                w.append(w_temp)

            self.predicted = np.vstack(predicted)
            self.l1 = np.vstack(l1)
            self.l2 = np.vstack(l2)
            self.l3 = np.vstack(l3)
            self.w = np.vstack(w)

            self.y_pred = self.predicted
            self.y_test = self.Y

    def Ensemble_Inference(self):
        models = os.listdir(os.path.join(self.Name,'models'))
        predicted = []
        w = []
        for model in models:
            self.Inference(model=model)
            predicted.append(self.predicted)
            w.append(self.w)

        predicted_dist = []
        for p in predicted:
            predicted_dist.append(np.expand_dims(p,0))
        predicted_dist = np.vstack(predicted_dist)

        w_dist = []
        for w_temp in w:
            w_dist.append(np.expand_dims(w_temp,0))
        w_dist = np.vstack(w_dist)

        self.predicted = np.mean(predicted_dist,0)
        self.predicted_dist = predicted_dist
        self.w = np.mean(w_dist,0)

        self.y_pred = self.predicted
        self.y_test = self.Y

    def Sample_Summary(self):
        self.Cell_Pred.sort_index(inplace=True)
        self.Cell_Pred['Patient'] = self.patients
        if hasattr(self,'predicted_dist'):
            group_dict = {'Label':'first'}
            for ii in self.lb.classes_:
                group_dict[ii] = 'mean'
                group_dict[ii+'_ci'] = 'mean'

        else:
            group_dict = {'Label':'first'}
            for ii in self.lb.classes_:
                group_dict[ii] = 'mean'

        self.sample_summary = self.Cell_Pred.groupby(['Patient']).agg(group_dict)

    def Sample_AUC_Curve(self):
        plt.figure()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        ROC_DFs = []
        for ii in self.lb.classes_:
            y_pred = np.asarray(self.sample_summary[ii])
            y_test = np.asarray(self.sample_summary['Label'])==ii
            roc_score = roc_auc_score(y_test, y_pred)
            fpr, tpr, th = roc_curve(y_test, y_pred)
            roc_df = pd.DataFrame()
            roc_df['th'] = th
            roc_df['fpr'] = fpr
            roc_df['tpr'] = tpr
            ROC_DFs.append(roc_df)
            plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (ii, roc_score))
        plt.legend(loc="lower right")
        plt.show()

        self.ROC_sample = dict(zip(self.lb.classes_,ROC_DFs))









