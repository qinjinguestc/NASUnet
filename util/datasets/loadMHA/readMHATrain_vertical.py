import os
import SimpleITK as sitk
import numpy as np
import re
import shutil


# path = 'F:\dataset\BRATS2015\Training\HGG'
# path1 = 'F:\dataset\BRATS2015\Training\LGG'

path = "E:\BRATS2015\BRATS2015_Training\HGG"
path1 = "E:\BRATS2015\BRATS2015_Training\LGG"

# path = "/media/zlz424/experiment/BRATS2015/BRATS2015_Training/HGG"
# path1 = "/media/zlz424/experiment/BRATS2015/BRATS2015_Training/LGG"


mha_save_path = r"./mha_save_path/"
mha_ground_truth = mha_save_path + "mha_ground_truth/"
mha_result = mha_save_path + "mha_result/"


class BRATS2015:
    def __init__(self, test_batch_count=0, train_batch_count=0):
        L = os.listdir(path)
        L1 = os.listdir(path1)
        self.OT_path = []
        self.Flair_path = []
        self.T1_path = []
        self.T1c_path = []
        self.T2_path = []
        self.test_batch_count = test_batch_count
        self.train_batch_count = train_batch_count

        for i in L:
            path_1 = path+'/'+i
            list_path = os.listdir(path_1)
            # print(list_path)
            for content in list_path:
                if "OT" in content:
                    path_2 = path_1+'/'+content
                    list_path2=os.listdir(path_2)
                    for content2 in list_path2:
                        if "OT" in content2:
                            path_3 = path_2 + '/' + content2
                            self.OT_path.append(path_3)
                if "T1." in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "T1." in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.T1_path.append(path_3)
                if "T1c" in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "T1c" in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.T1c_path.append(path_3)
                if "T2" in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "T2" in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.T2_path.append(path_3)
                if "Flair" in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "Flair" in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.Flair_path.append(path_3)
        for i in L1:
            path_1 = path1 + '/' + i  # into brats_2013_pat0001_*
            list_path = os.listdir(path_1)
            # print(list_path)
            for content in list_path:  # into VSD folders
                if "OT" in content:
                    path_2 = path_1 + '/' + content
                    list_path2 = os.listdir(path_2)
                    for content2 in list_path2:
                        if "OT" in content2:
                            path_3 = path_2 + '/' + content2
                            self.OT_path.append(path_3)
                if "T1." in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "T1." in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.T1_path.append(path_3)
                if "T1c" in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "T1c" in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.T1c_path.append(path_3)
                if "T2" in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "T2" in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.T2_path.append(path_3)
                if "Flair" in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "Flair" in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.Flair_path.append(path_3)

        self.total_batch = len(self.OT_path)
        self.test_batch = 30
        self.train_batch = self.total_batch - self.test_batch
        # print(self.train_batch,self.test_batch)
        self.next_train_MHA()
        self.train_batch_index = 0
        self.test_batch_index = 0
        self.saveArr = None
        self.next_test_MHA()

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
        self.saveitk_ot = ot # 'VSD.InstanceFCN' + re.findall(r"\.\d+\.", ot)[0] + 'mha'

        # 横切变纵切
        img_array = self.__readimg__(ot, isot=True).reshape((-1, 155, 240), order='C')
        img_array_t1 = self.__readimg__(t1).reshape((-1, 155, 240), order='C')
        img_array_t2 = self.__readimg__(t2).reshape((-1, 155, 240), order='C')
        img_array_t1c = self.__readimg__(t1c).reshape((-1, 155, 240), order='C')
        img_array_flair = self.__readimg__(flair).reshape((-1, 155, 240), order='C')

        arr_ot = np.asarray(img_array)
        # num, height, weight = np.shape(arr_ot)
        self.arr_test_ot = arr_ot#.reshape(num, height, weight)
        arr_t1 = np.expand_dims(img_array_t1, -1) #np.asarray(img_array_t1).reshape(num, height, weight, 1)
        arr_t1c = np.expand_dims(img_array_t1c, -1) #np.asarray(img_array_t1c).reshape(num, height, weight, 1)
        arr_t2 = np.expand_dims(img_array_t2, -1) #np.asarray(img_array_t2).reshape(num, height, weight, 1)
        arr_flair = np.expand_dims(img_array_flair, -1) #np.asarray(img_array_flair).reshape(num, height, weight, 1)
        self.arr_test_imgs = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=3)
        shape = self.arr_test_imgs.shape
        # self.arr_test_imgs = self.arr_test_imgs.reshape(-1, shape[3])
        # self.standard_scaler.fit(self.arr_train_imgs)
        # Max_min = preprocessing.MinMaxScaler()
        # self.arr_test_imgs = preprocessing.StandardScaler().fit_transform(self.arr_test_imgs)
        #print('next_test_MHA shape :'+shape)
        self.arr_test_imgs = self.arr_test_imgs.reshape(shape[0], shape[1], shape[2], shape[3])

        self.test_batch_count += 1

    def next_test_batch(self, batch_size):
        endbatch = np.shape(self.arr_test_ot)[0]
        # print('endbatch'+self.test_batch_index + batch_size, endbatch)
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

        # 横切变纵切
        img_array = self.__readimg__(ot, isot=True).transpose(1, 0, 2)
        img_array_t1 = self.__readimg__(t1).transpose(1, 0, 2)
        img_array_t2 = self.__readimg__(t2).transpose(1, 0, 2)
        img_array_t1c = self.__readimg__(t1c).transpose(1, 0, 2)
        img_array_flair = self.__readimg__(flair).transpose(1, 0, 2)

        arr_ot = np.asarray(img_array)

        # num, height, weight, chanel = np.shape(arr_ot)

        # index = np.asarray(range(num))
        # np.random.shuffle(index)
        #
        # arr_ot = arr_ot[index]
        # img_array_t1 = img_array_t1[index]
        # img_array_t2 = img_array_t2[index]
        # img_array_flair = img_array_flair[index]
        # img_array_t1c = img_array_t1c[index]

        self.arr_train_ot = arr_ot
        # self.arr_train_ot[self.arr_train_ot == 0] = -1
        arr_t1 = np.expand_dims(img_array_t1, -1)
        arr_t1c = np.expand_dims(img_array_t1c, -1)
        arr_t2 = np.expand_dims(img_array_t2, -1)
        arr_flair = np.expand_dims(img_array_flair, -1)

        self.arr_train_imgs = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=3)

        # 只训练有肿瘤的部分
        # has = np.max(self.arr_train_ot, axis=(1, 2)) > 0
        # self.arr_train_ot = self.arr_train_ot[has]
        # self.arr_train_imgs = self.arr_train_imgs[has]

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
            # print(np.shape(self.saveArr))
            img = sitk.GetImageFromArray(self.saveArr.reshape((155, 240, 240), order='C'))
            # img = np.asarray(img).reshape((155, 240, 240), order='C')
            path = self.OT_path[self.test_batch_count + self.train_batch - 2]
            # train：path = self.OT_path[self.train_batch_count - 2]
            name = 'VSD.DenseUnet' + re.findall(r"\.\d+\.", path)[0] + 'mha'
            print("saveITK", name)
            if not os.path.exists(mha_ground_truth):
                os.makedirs(mha_ground_truth)
            if not os.path.exists(mha_result):
                os.makedirs(mha_result)
            shutil.copy(path, mha_ground_truth + name)
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), os.path.join(mha_result, name))
            self.saveArr = None
