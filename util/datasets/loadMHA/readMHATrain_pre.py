import os
import SimpleITK as sitk
import numpy as np
import re
import shutil
from glob import glob


path = 'F:\dataset\BRATS2015\Training\HGG'
path1 = 'F:\dataset\BRATS2015\Training\LGG'

# path = "/media/zlz424/experiment/BRATS2015/brats_n4_bias_training/HGG"
# path1 = "/media/zlz424/experiment/BRATS2015/brats_n4_bias_training/LGG"

# path = "/media/dengerqiang/B8621E80621E4392/WDY_experiment/nbisBrat2015/brats_n4_bias_training/HGG"
# path1 = "/media/dengerqiang/B8621E80621E4392/WDY_experiment/nbisBrat2015/brats_n4_bias_training/LGG"

# path = r"E:\BRATS2015\brats_n4_bias_training\HGG"
# path1 = r"E:\BRATS2015\brats_n4_bias_training\LGG"

# path = 'E:\Dataset\Brats\BRATS2015\Training\HGG'
# path1 = 'E:\Dataset\Brats\BRATS2015\Training\LGG'

mha_save_path = r"/media/zlz424/Program/wdy_experiment/DenseUnet/mha_save_path/"
mha_ground_truth = mha_save_path + "mha_ground_truth/"
mha_result = mha_save_path + "mha_result/"


class BRATS2015:
    def __init__(self, test_batch_count=0, train_batch_count=0):
        L = os.listdir(path)
        L1 = os.listdir(path1)
        L.extend(L1)
        self.OT_path = []
        self.Flair_path = []
        self.T1_path = []
        self.T1c_path = []
        self.T2_path = []
        self.test_batch_count = test_batch_count
        self.train_batch_count = train_batch_count
        num = 0
        for i in L:
            num += 1
            if num <= 220:
                path_1 = path + '/' + i  # into brats_2013_pat0001_*
                OT_path = glob(path_1 + '/*OT*.mha')
                T1_path = glob(path_1 + '/*T1.*.mha')
                T1c_path = glob(path_1 + '/*T1c*.mha')
                Flair_path = glob(path_1 + '/*Flair*.mha')
                T2_path = glob(path_1 + '/*T2*.mha')
                self.OT_path.append(OT_path[0])
                self.T1_path.append(T1_path[0])
                self.T1c_path.append(T1c_path[0])
                self.T2_path.append(T2_path[0])
                self.Flair_path.append(Flair_path[0])
            else:
                path_1 = path1 + '/' + i  # into brats_2013_pat0001_*
                OT_path = glob(path_1 + '/*OT*.mha')
                T1_path = glob(path_1 + '/*T1.*.mha')
                T1c_path = glob(path_1 + '/*T1c*.mha')
                Flair_path = glob(path_1 + '/*Flair*.mha')
                T2_path = glob(path_1 + '/*T2*.mha')
                self.OT_path.append(OT_path[0])
                self.T1_path.append(T1_path[0])
                self.T1c_path.append(T1c_path[0])
                self.T2_path.append(T2_path[0])
                self.Flair_path.append(Flair_path[0])
        self.total_batch = len(self.OT_path)
        self.test_batch = 0
        self.train_batch = self.total_batch - self.test_batch
        # print(self.train_batch,self.test_batch)
        self.next_train_MHA()
        self.train_batch_index = 0
        self.saveArr = None
        # self.next_test_MHA()
        # self.test_batch_index = 0

    def __readimg__(self, mha_path, isot=False):
        mha = sitk.ReadImage(mha_path)
        img_array = sitk.GetArrayFromImage(mha)
        if isot:
            return img_array
        else:
            img_array = (img_array - img_array.mean()) / img_array.std()
            return img_array

    def __read_img_test__(self, mha_path, isot=False):
        mha = sitk.ReadImage(mha_path)
        img_array = sitk.GetArrayFromImage(mha)
        if isot:
            return img_array
        else:
            img_array = (img_array - img_array.mean()) / img_array.std()
            return img_array

    def next_test_MHA(self):

        self.test_batch_count = self.test_batch_count #% self.test_batch

        ot = self.OT_path[self.test_batch_count % self.test_batch + self.train_batch]
        t1 = self.T1_path[self.test_batch_count % self.test_batch + self.train_batch]
        t2 = self.T2_path[self.test_batch_count % self.test_batch + self.train_batch]
        t1c = self.T1c_path[self.test_batch_count % self.test_batch + self.train_batch]
        flair = self.Flair_path[self.test_batch_count % self.test_batch + self.train_batch]
        #print('VSD.InstanceFCN' + re.findall(r"\.\d+\.", ot)[0] + 'mha')
        self.saveitk_ot = ot  # 'VSD.InstanceFCN' + re.findall(r"\.\d+\.", ot)[0] + 'mha'

        self.arr_test_ot = self.__read_img_test__(ot, isot=True)
        img_array_t1 = self.__read_img_test__(t1)
        img_array_t2 = self.__read_img_test__(t2)
        img_array_t1c = self.__read_img_test__(t1c)
        img_array_flair = self.__read_img_test__(flair)

        arr_t1 = np.expand_dims(img_array_t1, -1)
        arr_t1c = np.expand_dims(img_array_t1c, -1)
        arr_t2 = np.expand_dims(img_array_t2, -1)
        arr_flair = np.expand_dims(img_array_flair, -1)
        self.arr_test_imgs = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=3)
        # self.arr_test_imgs = self.arr_test_imgs.reshape(-1, shape[3])
        # self.standard_scaler.fit(self.arr_train_imgs)
        # Max_min = preprocessing.MinMaxScaler()
        # self.arr_test_imgs = preprocessing.StandardScaler().fit_transform(self.arr_test_imgs)
        #print('next_test_MHA shape :'+shape)
        self.test_batch_count += 1

    def next_test_batch(self, batch_size):
        endbatch = np.shape(self.arr_test_ot)[0]
        if self.test_batch_index + batch_size >= endbatch:
            img = self.arr_test_imgs[self.test_batch_index:endbatch]
            label = self.arr_test_ot[self.test_batch_index:endbatch]
            self.test_batch_index = 0
            self.next_test_MHA()
        else:
            img = self.arr_test_imgs[self.test_batch_index:self.test_batch_index + batch_size]
            label = self.arr_test_ot[self.test_batch_index:self.test_batch_index + batch_size]
            self.test_batch_index += batch_size
        # label = np.eye(5)[label]
        # label[:, :, :, 0] = label[:, :, :, 0] * 0.05
        return img, label  # , makelabel_tf(label)
            # return img, label

    def next_train_MHA(self):
        # if self.train_batch_count > self.train_batch:
        self.train_batch_count = self.train_batch_count % self.train_batch
        # print(self.OT_path[self.train_batch_count])
        ot = self.OT_path[self.train_batch_count]
        t1 = self.T1_path[self.train_batch_count]
        t2 = self.T2_path[self.train_batch_count]
        t1c = self.T1c_path[self.train_batch_count]
        flair = self.Flair_path[self.train_batch_count]

        arr_ot = self.__readimg__(ot, isot=True)
        img_array_t1 = self.__readimg__(t1)
        img_array_t2 = self.__readimg__(t2)
        img_array_t1c = self.__readimg__(t1c)
        img_array_flair = self.__readimg__(flair)

        #打乱顺序
        num, height, weight = np.shape(arr_ot)
        index = np.asarray(range(num))
        np.random.shuffle(index)
        arr_ot = arr_ot[index]
        img_array_t1 = img_array_t1[index]
        img_array_t2 = img_array_t2[index]
        img_array_flair = img_array_flair[index]
        img_array_t1c = img_array_t1c[index]

        self.arr_train_ot = arr_ot
        # self.arr_train_ot[self.arr_train_ot == 0] = -1
        arr_t1 = np.expand_dims(img_array_t1, -1)
        arr_t1c = np.expand_dims(img_array_t1c, -1)
        arr_t2 = np.expand_dims(img_array_t2, -1)
        arr_flair = np.expand_dims(img_array_flair, -1)
        self.arr_train_imgs = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=-1)

        # 只训练有肿瘤的部分
        has = np.max(self.arr_train_ot, axis=(1, 2)) > 0
        self.arr_train_ot = self.arr_train_ot[has]
        self.arr_train_imgs = self.arr_train_imgs[has]

        # 图像分到有肿瘤大小
        # has = np.where(self.arr_train_ot != 0)
        # z_min, x_min, y_min = np.min(has, -1)
        # z_max, x_max, y_max = np.max(has, -1)

        # # print(x_min, x_max, y_min, y_max)

        # x_max = 8 - (x_max - x_min) % 8 + x_max
        # y_max = 8 - (y_max - y_min) % 8 + y_max
        # # print(x_min, x_max, y_min, y_max)

        # self.arr_train_imgs = self.arr_train_imgs[z_min:z_max, x_min:x_max, y_min:y_max, :]
        # self.arr_train_ot = self.arr_train_ot[z_min:z_max, x_min:x_max, y_min:y_max]

        # index = np.asarray(range(len(self.arr_train_imgs)))
        # np.random.shuffle(index)

        # self.arr_train_ot[index]
        # self.arr_train_imgs[index]

        # print(self.arr_train_imgs.shape)
        self.train_batch_count += 1

    def next_train_batch(self, batch_size):
        endbatch = np.shape(self.arr_train_ot)[0]
        # print(self.train_batch_index + batch_size, endbatch)
        if self.train_batch_index + batch_size >= endbatch:
            img = self.arr_train_imgs[self.train_batch_index:endbatch]
            label = self.arr_train_ot[self.train_batch_index:endbatch]
            self.train_batch_index = 0
            self.next_train_MHA()

        else:
            img = self.arr_train_imgs[self.train_batch_index:self.train_batch_index + batch_size]
            label = self.arr_train_ot[self.train_batch_index:self.train_batch_index + batch_size]
            self.train_batch_index += batch_size

        # label = np.eye(5)[label]
        # label[:,:,:,0] = label[:,:,:,0] * 0.01
        # class_weight[:,:,:,0] = class_weight[:,:,:,0] * 0.1

        # label[label > 0] = 1
        # label = np.eye(2)[label]
        return img, label

    def saveItk(self, array):
        array = np.asarray(array)
        if self.saveArr is not None:
            self.saveArr = np.concatenate([self.saveArr, array], axis=0)
        else:
            self.saveArr = array
        if self.test_batch_index == 0:
            img = sitk.GetImageFromArray(self.saveArr)
            path = self.OT_path[self.test_batch_count + self.train_batch - 2]
            # train：path = self.OT_path[self.train_batch_count - 2]
            name = 'VSD.DenseUnet_pre' + re.findall(r"\.\d+\.", path)[0] + 'mha'
            print("saveITK", name)
            if not os.path.exists(mha_ground_truth):
                os.makedirs(mha_ground_truth)
            if not os.path.exists(mha_result):
                os.makedirs(mha_result)
            shutil.copy(path, mha_ground_truth + name)
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), os.path.join(mha_result, name))
            self.saveArr = None

# brats =  BRATS2015()
# i = brats.arr_train_imgs#= brats.next_train_batch(1)
#
#
# import matplotlib.pyplot as plt
#
# for nnn in range(50):
#     i1 = i[nnn,:,:,0]
#     sub = plt.subplot(221)
#     sub.imshow(i1, cmap='gray')
#     i1 = i[nnn,:,:,1]
#     sub = plt.subplot(222)
#     sub.imshow(i1, cmap='gray')
#     i1 = i[nnn,:,:,2]
#     sub = plt.subplot(223)
#     sub.imshow(i1, cmap='gray')
#     i1 = i[nnn,:,:,3]
#     sub = plt.subplot(224)
#     sub.imshow(i1, cmap='gray')
#     plt.show()
# a = []
# for i in range(190):
#    l = brats.arr_train_ot
#    print(np.sum(l,(0,1,2)))
#    l = np.eye(5)[l]
#    # print(np.sum(l,(3)))
#    print(np.sum(l,(0,1,2)))
#    a.append(np.sum(l, (0,1,2)))
#    brats.next_train_MHA()
#
# a = np.asarray(a)
# print(a.sum(0))

#
# label_to_frequency =  [  1.67481864e+09,   1.05296600e+06,   1.39822550e+07,   2.25209600e+06,   4.21404400e+06]
# class_weights = []
#
# total_frequency = np.sum(label_to_frequency)
# for frequency in label_to_frequency:
#     class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
#     class_weights.append(class_weight)
# print(class_weights)

# [1.4496326123173364, 47.438213320786453, 27.297160306177677, 44.379722047746604, 40.152262287301141]
