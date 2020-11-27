import os
from glob import glob

import SimpleITK as sitk
import numpy as np
import re
# from sklearn import preprocessing
# from sklearn.externals import joblib
import shutil


# path = "/media/dengerqiang/B8621E80621E4392/WDY_experiment/nbisBrat2015/brats_n4_bias_testing"
# path = r'E:\BRATS2015\brats_n4_bias_testing'
# path = r'E:\BRATS2015\BRATS2015_Testing\HGG_LGG'
path = r'H:\Dataset\BRATS2015\Testing\HGG_LGG'


def makelabel_tf(label):
    label = np.reshape(label,[np.shape(label)[0], -1])
    l = np.max(label,axis=1)
    one = np.asarray([1 if i > 0 else 0 for i in l])
    return one


mha_save_path = "./mha_save_path/"
mha_ground_truth = mha_save_path + "mha_ground_truth/"
mha_result = mha_save_path + "mha_result/"


class BRATS2015:
    def __init__(self, test_batch_count=0):
        L = os.listdir(path)
        self.OT_path = []
        self.Flair_path = []
        self.T1_path = []
        self.T1c_path = []
        self.T2_path = []
        self.test_batch_count = test_batch_count
        if "n4" in path:
            for i in L:
                path_1 = path + '/' + i  # into brats_2013_pat0001_*
                T1_path = glob(path_1 + '/*T1.*.mha')
                T1c_path = glob(path_1 + '/*T1c*.mha')
                Flair_path = glob(path_1 + '/*Flair*.mha')
                T2_path = glob(path_1 + '/*T2*.mha')
                self.T1_path.append(T1_path[0])
                self.T1c_path.append(T1c_path[0])
                self.T2_path.append(T2_path[0])
                self.Flair_path.append(Flair_path[0])
        else:
            for i in L:
                path_1 = path + '/' + i  # into brats_2013_pat0001_*
                T1_path = glob(path_1 + '/*/*T1.*.mha')
                T1c_path = glob(path_1 + '/*/*T1c*.mha')
                Flair_path = glob(path_1 + '/*/*Flair*.mha')
                T2_path = glob(path_1 + '/*/*T2*.mha')
                self.T1_path.append(T1_path[0])
                self.T1c_path.append(T1c_path[0])
                self.T2_path.append(T2_path[0])
                self.Flair_path.append(Flair_path[0])

        self.test_batch = len(self.T1_path)
        self.test_batch_index = 0
        self.saveArr = None
        self.next_test_MHA()

    # def __read_img_test__(self, mha_path, isot=False):
    #     mha = sitk.ReadImage(mha_path)
    #     img_array = sitk.GetArrayFromImage(mha)
    #     # n, h, w = img_array.shape
    #     img_array = (img_array - img_array.mean()) / img_array.std()
    #     img_array = np.expand_dims(img_array, -1)
    #     # X_train_minmax = img_array.reshape(n, w, h, 1)
    #     return img_array

    def __read_img_test__(self, mha_path, isot=False):
        mha = sitk.ReadImage(mha_path)
        img_array = sitk.GetArrayFromImage(mha)
        if isot:
            return img_array
        # else:
        #     img_array = (img_array - img_array.mean()) / img_array.std()
        #     return img_array
        else:
            mask = img_array > 0
            temp_img = img_array[img_array > 0]
            # img_array = (img_array-temp_img.mean())/temp_img.std()
            # img_array = (img_array-img_array.min())/(img_array.max()-img_array.min())*mask

            img_array = (img_array - temp_img.mean()) / temp_img.std() * mask

        return img_array

    def next_test_MHA(self):
        t1 = self.T1_path[self.test_batch_count % self.test_batch]
        t2 = self.T2_path[self.test_batch_count % self.test_batch]
        t1c = self.T1c_path[self.test_batch_count % self.test_batch]
        flair = self.Flair_path[self.test_batch_count % self.test_batch]

        # img_array = self.__read_img_test__(ot, isot=True)
        img_array_t1 = self.__read_img_test__(t1)
        img_array_t2 = self.__read_img_test__(t2)
        img_array_t1c = self.__read_img_test__(t1c)
        img_array_flair = self.__read_img_test__(flair)

        # arr_ot = np.asarray(img_array)
        # num, height, weight, chanel = np.shape(img_array_t1)

        # self.arr_test_ot = arr_ot.reshape(num, height, weight)
        # self.arr_test_ot[self.arr_test_ot == 0] = -1
        arr_t1 = np.expand_dims(img_array_t1, -1)
        arr_t1c = np.expand_dims(img_array_t1c, -1)
        arr_t2 = np.expand_dims(img_array_t2, -1)
        arr_flair = np.expand_dims(img_array_flair, -1)

        self.arr_test_imgs = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=3)
        shape = self.arr_test_imgs.shape
        self.arr_test_imgs = self.arr_test_imgs.reshape(-1, shape[3])
        self.arr_test_imgs = self.arr_test_imgs.reshape(shape[0], shape[1], shape[2], shape[3])
        self.test_batch_count += 1

    def next_test_batch(self, batch_size):
        endbatch = np.shape(self.arr_test_imgs)[0]
        # print(self.test_batch_index + batch_size, endbatch)
        if self.test_batch_index + batch_size >= endbatch:
            img = self.arr_test_imgs[self.test_batch_index:endbatch]
            self.test_batch_index = 0
            self.next_test_MHA()
        else:
            img = self.arr_test_imgs[self.test_batch_index:self.test_batch_index + batch_size]
            self.test_batch_index += batch_size
        return img

    def saveItk(self, array):
        array = np.asarray(array)
        if self.saveArr is not None:
            self.saveArr = np.concatenate([self.saveArr, array], axis=0)
        else:
            self.saveArr = array
        if self.test_batch_index == 0:
            img = sitk.GetImageFromArray(self.saveArr)
            path = self.Flair_path[self.test_batch_count - 2]
            name = 'VSD.DENSEUNET_test' + re.findall(r"\.\d+\.", path)[0] + 'mha'
            print(name, self.test_batch_count-1)
            if not os.path.exists(mha_result):
                os.makedirs(mha_result)
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), os.path.join(mha_result, name))
            self.saveArr = None

# brats = BRATS2015()

# def make_else_brain_label(img, label):
#     m = img[:, :, :, 0] > 0
#     n = np.zeros(m.shape)
#     n[m] = 1
#     # print(n)
#     n[label[:,:,:] !=0] = 0
#     label[n ==1] = 5
#     return img, label
#
# def make_tumor_label(img, label):
#     label[label > 0] = 1
#     return img, label
# i,l = brats.next_train_batch(10)
# l =make_else_brain_label(i,l)
# plt.imshow(l[0])
# print(l.shape)
# plt.show()
# print(10*240*240)
# print(np.eye(*5)[l].shape)
# for i in range(190):
    # a = np.max(brats.arr_train_ot, axis=(1,2)) > 0
    # c = np.max(brats.arr_train_ot, axis=(1,2)) == 0
    # # print(c)arr_train_ot
    # has = brats.arr_train_ot[a]
    # nothave = brats.arr_train_ot[c]
    # print(has.shape,nothave.shape)
    # brats.next_train_MHA()


# npy_path = 'D:/brats_npy/'
# brats = BRATS2015()
# for i in range(190):
#     for j in range(155):
#         img, label = brats.next_train_batch(1)
#         np.save(npy_path+'img/'+str(i)+'_'+str(j)+'.npy', img)
#         np.save(npy_path+'label/'+str(i)+'_'+str(j)+'.npy', label)
#         print(i, j)


# brats = BRATS2015Test()
# # for i in range(200):
# print(brats.test_batch_count)
# brats.next_test_MHA()
# print(brats.test_batch_count)
# brats.next_test_MHA()
# print(brats.test_batch_count)
# print(brats.next_test_batch(5).shape)


