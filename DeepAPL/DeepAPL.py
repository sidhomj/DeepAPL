import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import os
import glob
import pickle
from DeepAPL.functions.utils import *
from DeepAPL.functions.Layers import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from copy import deepcopy
import shutil

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

            pts_exclude = []
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
                        pts_exclude.append(pt)


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

    def _reset_models(self):
        self.models_dir = os.path.join(self.Name,'models')
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)
        os.makedirs(self.models_dir)

    def _build(self,weight_by_class=False,multisample_dropout_num_masks = None,graph_seed=None):
        GO = graph_object()
        with tf.device(self.device):
            GO.graph_model = tf.Graph()
            with GO.graph_model.as_default():
                if graph_seed is not None:
                    tf.set_random_seed(graph_seed)
                Get_Inputs(GO,self)
                Conv_Model(GO)
                if multisample_dropout_num_masks is not None:
                    GO.w = MultiSample_Dropout(X=GO.l3,
                                               num_masks=multisample_dropout_num_masks,
                                               units=GO.Y.shape[1],
                                               rate=GO.prob_multisample,
                                               activation=None)
                else:
                    GO.w = tf.layers.dense(GO.l3, GO.Y.shape[1])
                GO.w = tf.identity(GO.w,'w')
                GO.logits = tf.reduce_mean(GO.w, [1, 2])

                if weight_by_class:
                    class_weights = tf.constant([(1 / (np.sum(self.Y, 0) / np.sum(self.Y))).tolist()])
                    weights = tf.squeeze(tf.matmul(tf.cast(GO.Y, dtype='float32'), class_weights, transpose_b=True), axis=1)
                else:
                    weights = 1

                GO.loss = tf.reduce_mean(weights*tf.nn.softmax_cross_entropy_with_logits_v2(GO.Y,GO.logits))

                GO.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(GO.loss)

                with tf.name_scope('Accuracy_Measurements'):
                    GO.predicted = tf.nn.softmax(GO.logits, name='predicted')
                    correct_pred = tf.equal(tf.argmax(GO.predicted, 1), tf.argmax(GO.Y, 1))
                    GO.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

                GO.saver = tf.train.Saver(max_to_keep=None)

        self.GO = GO

    def _train(self,batch_size = 10, epochs_min = 10,stop_criterion=0.001,stop_criterion_window=10,
               dropout_rate=0.0,multisample_dropout_rate=0.0,iteration=0):
        GO = self.GO
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        training = True
        e=1
        stop_check_list = []
        val_loss_total = []
        with tf.Session(graph=GO.graph_model, config=config) as sess:
            sess.run(tf.global_variables_initializer())
            while training is True:
                Vars = [self.train[0],self.train[3]]
                loss_temp = []
                for vars in get_batches(Vars, batch_size=batch_size, random=True):
                    feed_dict = {GO.X: vars[0],
                                 GO.Y: vars[1],
                                 GO.prob:dropout_rate,
                                 GO.prob_multisample: multisample_dropout_rate}
                    loss_i,_  = sess.run([GO.loss,GO.opt],feed_dict=feed_dict)
                    loss_temp.append(loss_i)

                train_loss = np.mean(loss_temp)

                Vars = [self.valid[0],self.valid[3]]
                loss_temp = []
                for vars in get_batches(Vars, batch_size=batch_size, random=False):
                    feed_dict = {GO.X: vars[0],
                                 GO.Y: vars[1]}
                    loss_i  = sess.run(GO.loss,feed_dict=feed_dict)
                    loss_temp.append(loss_i)

                valid_loss = np.mean(loss_temp)
                val_loss_total.append(valid_loss)

                Vars = [self.test[0],self.test[3]]
                loss_temp = []
                pred_temp = []
                for vars in get_batches(Vars, batch_size=batch_size, random=False):
                    feed_dict = {GO.X: vars[0],
                                 GO.Y: vars[1]}
                    loss_i,pred_i  = sess.run([GO.loss,GO.predicted],feed_dict=feed_dict)
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

            #Get Activation Maps
            Vars = [self.imgs]
            w = []
            for vars in get_batches(Vars, batch_size=batch_size, random=False):
                feed_dict = {GO.X: vars[0]}
                w_temp = sess.run(GO.w,feed_dict=feed_dict)
                w.append(w_temp)

            self.w = np.vstack(w)
            self.predicted[self.test_idx] += self.y_pred
            GO.saver.save(sess, os.path.join(self.Name, 'models','model_'+str(iteration),'model.ckpt'))

    def Train(self,weight_by_class=False,multisample_dropout_num_masks = None,graph_seed=None,
              batch_size = 10, epochs_min = 10,stop_criterion=0.001,stop_criterion_window=10,
              dropout_rate=0.0,multisample_dropout_rate=0.0):
        self._reset_models()
        self._build(weight_by_class,multisample_dropout_num_masks,graph_seed)
        self._train(batch_size, epochs_min,stop_criterion,stop_criterion_window,
                    dropout_rate,multisample_dropout_rate)

    def Monte_Carlo_CrossVal(self,folds=5,seeds=None,test_size=0.25,combine_train_valid=False,train_all=False,
                             weight_by_class=False, multisample_dropout_num_masks=None,graph_seed=None,
                             batch_size=10, epochs_min=10, stop_criterion=0.001, stop_criterion_window=10,
                            dropout_rate = 0.0, multisample_dropout_rate = 0.0):

        y_pred = []
        y_test = []
        w = []
        predicted = np.zeros_like(self.predicted)
        counts = np.zeros_like(self.predicted)
        self._reset_models()
        self._build(weight_by_class,multisample_dropout_num_masks,graph_seed)

        for i in range(0, folds):
            print(i)
            if seeds is not None:
                np.random.seed(seeds[i])
            self.Get_Train_Valid_Test(test_size=test_size,combine_train_valid=combine_train_valid,train_all=train_all)
            self._train(batch_size, epochs_min, stop_criterion, stop_criterion_window,
                        dropout_rate, multisample_dropout_rate,iteration=i)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)
            w.append(self.w)

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
        self.counts = counts
        w_dist = []
        for w_temp in w:
            w_dist.append(np.expand_dims(w_temp,0))
        w_dist = np.vstack(w_dist)
        self.w = np.mean(w_dist,0)
        print('Monte Carlo Simulation Completed')

    def Get_Cell_Predicted(self,confidence=0.95,Load_Prev_Data=False):
        if Load_Prev_Data is False:
            df = pd.DataFrame()
            df['Patient'] = self.patients
            df['Cell_Type'] = self.cell_type
            df['Label'] = self.labels
            df['Files'] = self.files
            df['Counts'] = self.counts[:,0]

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

            with open(os.path.join(self.Name,'cell_preds.pkl'),'wb') as f:
                pickle.dump(df,f,protocol=4)
        else:
            with open(os.path.join(self.Name,'cell_preds.pkl'),'rb') as f:
                df = pickle.load(f)

        self.Cell_Pred = df

    def Sample_Summary(self,confidence=0.95,Load_Prev_Data=False):
        self.Get_Cell_Predicted(confidence,Load_Prev_Data)
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
        self.Sample_Summary()
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

    def Representative_Cells(self,type='APL',num=12,confidence=0.95,cell_type=None,Load_Prev_Data=False):
        self.Get_Cell_Predicted(confidence,Load_Prev_Data=Load_Prev_Data)
        df = deepcopy(self.Cell_Pred)
        if cell_type is not None:
            df = df[df['Cell_Type'] == cell_type]
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
            w_var = graph.get_tensor_by_name('w:0')

            predicted = []
            w = []
            for x in get_batches([self.imgs], batch_size=batch_size, random=False):
                feed_dict = {X: x[0]}
                predicted_i,w_temp = sess.run([pred,w_var],feed_dict=feed_dict)
                predicted.append(predicted_i)
                w.append(w_temp)

            self.predicted = np.vstack(predicted)
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










