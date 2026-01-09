from abc import ABCMeta, abstractmethod
import imageio.v2 as imageio
import imgaug as ia
from imgaug import augmenters as iaa
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image



class FileHandler:
    def CheckDir(self,folder_path):
        if os.path.exists(folder_path)==False:
            os.mkdir(folder_path)
            
    def ChangeRoot(self,folder,root):
        if len(root.split('\\',1))>1:
            return os.path.join(folder,root.split('\\',1)[1])
        else:
            return os.path.join(folder)

class ImageAugment(FileHandler,metaclass=ABCMeta):
    def __init__(self,Input_dir,Output_dir):
        self.Input_dir=Input_dir
        self.Output_dir=Output_dir
    
    @abstractmethod
    def Augment(self,aug_multi,img_file_extension,mode):
        pass
    
class CNN(ImageAugment):
    def __init__(self,Input_dir="./Original_Image",Output_dir="./Output_Image"):
        super().__init__(Input_dir, Output_dir)
       
    def Augment(self,aug_multi,img_file_extension,mode="Sequential"):
        for root, sub_folders, files in os.walk(self.Input_dir):
            for name in files:
                img=imageio.imread(os.path.join(super().ChangeRoot(self.Input_dir,root), name[:-4] + img_file_extension))
                if mode=="Sequential":
                    aug = iaa.Sequential([
                    iaa.Fliplr(), #水平翻轉
                    iaa.Flipud(), #垂直翻轉
                    iaa.Rot90(1), #旋轉 90 度
                    iaa.Rot90(3), #旋轉 270 度
                    iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)), #高斯噪聲
                    iaa.SaltAndPepper(0.1), #椒鹽噪聲
                    iaa.MultiplyBrightness((0.5, 1.5)), #倍增亮度
                    iaa.GammaContrast((0.5, 2.0)) #伽瑪對比度
                    ], random_order=True)
                elif mode=="SomeOf":
                    aug = iaa.SomeOf(2, [
                    iaa.Fliplr(), #水平翻轉
                    iaa.Flipud(), #垂直翻轉
                    iaa.Rot90(1), #旋轉 90 度
                    iaa.Rot90(3), #旋轉 270 度
                    iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)), #高斯噪聲
                    iaa.SaltAndPepper(0.1), #椒鹽噪聲
                    iaa.MultiplyBrightness((0.5, 1.5)), #倍增亮度
                    iaa.GammaContrast((0.5, 2.0)) #伽瑪對比度
                    ])
                elif mode=="OneOf":
                    aug = iaa.OneOf([
                    iaa.Fliplr(), #水平翻轉
                    iaa.Flipud(), #垂直翻轉
                    iaa.Rot90(1), #旋轉 90 度
                    iaa.Rot90(3), #旋轉 270 度
                    iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)), #高斯噪聲
                    iaa.SaltAndPepper(0.1), #椒鹽噪聲
                    iaa.MultiplyBrightness((0.5, 1.5)), #倍增亮度
                    iaa.GammaContrast((0.5, 2.0)) #伽瑪對比度
                    ])
                images_aug = [aug.augment_image(img) for _ in range(aug_multi)]
                super().CheckDir(super().ChangeRoot(self.Output_dir,root))
                for index,image in enumerate(images_aug):
                    imageio.imwrite(os.path.join(super().ChangeRoot(self.Output_dir,root),str(name[:-4]) + "_" + str(index) + img_file_extension),image)
   
class OD(ImageAugment):
    def __init__(self,Input_dir="./Original_Image",Output_dir="./Output_Image",Annotation_Input_dir="./Annotation_File"):
        super().__init__(Input_dir, Output_dir)
        self.Annotation_Input_dir=Annotation_Input_dir
               
    def Read_xml_annotation(self,root, image_id):
        in_file = open(os.path.join(root, image_id))
        tree = ET.parse(in_file)
        root = tree.getroot()
        bndboxlist = []
        for object in root.findall('object'):  # 找到root節點下的所有country節點
            bndbox = object.find('bndbox')  # 子節點下節點rank的值
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            # print(xmin,ymin,xmax,ymax)
            bndboxlist.append([xmin,ymin,xmax,ymax])
            # print(bndboxlist)
        bndbox = root.find('object').find('bndbox')
        return bndboxlist # 以多維數組的形式保存

    def Change_xml_list_annotation(self,root, image_id, new_w, new_h, new_target,saveroot,id):
        in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 讀取原來的xml文件
        tree = ET.parse(in_file) # 讀取xml文件
        xmlroot = tree.getroot()
        w=xmlroot.find('size').find('width')
        w.text=str(new_w)
        h=xmlroot.find('size').find('height')
        h.text=str(new_h)
        index = 0
        # 將bbox中原來的座標值換成新生成的座標值
        for object in xmlroot.findall('object'):  # 找到root節點下的所有country節點
            bndbox = object.find('bndbox')  # 子節點下節點rank的值
            # xmin = int(bndbox.find('xmin').text)
            # xmax = int(bndbox.find('xmax').text)
            # ymin = int(bndbox.find('ymin').text)
            # ymax = int(bndbox.find('ymax').text)
            # 注意new_target原本保存為高維數組
            new_xmin = new_target[index][0]
            new_ymin = new_target[index][1]
            new_xmax = new_target[index][2]
            new_ymax = new_target[index][3]
            xmin = bndbox.find('xmin')
            xmin.text = str(new_xmin)
            ymin = bndbox.find('ymin')
            ymin.text = str(new_ymin)
            xmax = bndbox.find('xmax')
            xmax.text = str(new_xmax)
            ymax = bndbox.find('ymax')
            ymax.text = str(new_ymax)
            index = index + 1
        tree.write(os.path.join(saveroot, str(image_id) + "_" + str(id) + '.xml'))

    def Read_txt_annotation(self,root, image_id, img_file_extension):
        with open(os.path.join(root, image_id),"r",encoding="utf-8") as f:
            bndboxlist = []
            img = Image.open(os.path.join(super().ChangeRoot(self.Input_dir,root), os.path.splitext(image_id)[0]+img_file_extension))
            w=img.width 
            h=img.height 
            lines = f.readlines()
            for line in lines:
                X_center, Y_center, Yolo_w, Yolo_h = float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4])
                xmin = int(w*(2*X_center-Yolo_w)/2)
                xmax = int(w*(2*X_center+Yolo_w)/2)
                ymin = int(h*(2*Y_center-Yolo_h)/2)
                ymax = int(h*(2*Y_center+Yolo_h)/2)
                bndboxlist.append([xmin,ymin,xmax,ymax])
            return bndboxlist

    def Change_txt_list_annotation(self,root, image_id, new_w, new_h, new_target,saveroot, id, img_file_extension):
        file_data = ""
        with open(os.path.join(root, str(image_id) + '.txt'),"r",encoding="utf-8") as f:
            w=new_w
            h=new_h
            lines = f.readlines()
            index = 0
            for line in lines:
                new_xmin = new_target[index][0]
                new_ymin = new_target[index][1]
                new_xmax = new_target[index][2]
                new_ymax = new_target[index][3]
                X_center = round((new_xmin+new_xmax)/2/w,6)
                Y_center = round((new_ymin+new_ymax)/2/h,6)
                Yolo_w = round((new_xmax-new_xmin)/w,6)
                Yolo_h = round((new_ymax-new_ymin)/h,6)
                index = index + 1
                new_line=line.split(' ',1)[0]+' '+str(X_center)+' '+str(Y_center)+' '+str(Yolo_w)+' '+str(Yolo_h)+"\n"
                file_data += new_line
        with open(os.path.join(saveroot, str(image_id) + "_" + str(id) + '.txt'),"w",encoding="utf-8") as f:
            f.write(file_data)  
    
    def Augment(self, aug_multi,img_file_extension,mode="Sequential",format_mode="xml"):
        super().CheckDir(self.Output_dir)
        boxes_img_aug_list = []
        new_bndbox_list = []
        if mode=="Sequential":
            aug = iaa.Sequential([
            iaa.ShearY((-20, 20)),  # vertically flip 20% of all images
            iaa.Fliplr(0.5),  # 鏡像
            iaa.Affine(
                translate_px={"x": 15, "y": 15},
                scale=(0.8, 0.95),
                rotate=(-30, 30)
            ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            iaa.Fliplr(), #水平翻轉
            iaa.Flipud(), #垂直翻轉
            iaa.Rot90(1), #旋轉 90 度
            iaa.Rot90(3), #旋轉 270 度
            iaa.AddToBrightness((-50, 50)),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.Resize({"height": 700, "width": 300})
            ])
        elif mode=="SomeOf":
            aug = iaa.SomeOf(2, [
            iaa.ShearY((-20, 20)),  # vertically flip 20% of all images
            iaa.Fliplr(0.5),  # 鏡像
            iaa.Affine(
                translate_px={"x": 15, "y": 15},
                scale=(0.8, 0.95),
                rotate=(-30, 30)
            ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            iaa.Fliplr(), #水平翻轉
            iaa.Flipud(), #垂直翻轉
            iaa.Rot90(1), #旋轉 90 度
            iaa.Rot90(3), #旋轉 270 度
            iaa.AddToBrightness((-50, 50)),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.Resize({"height": 300, "width": 300})
            ])
        elif mode=="OneOf":
            aug = iaa.OneOf([
            iaa.ShearY((-20, 20)),  # vertically flip 20% of all images
            iaa.Fliplr(0.5),  # 鏡像
            iaa.Affine(
                translate_px={"x": 15, "y": 15},
                scale=(0.8, 0.95),
                rotate=(-30, 30)
            ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            iaa.Fliplr(), #水平翻轉
            iaa.Flipud(), #垂直翻轉
            iaa.Rot90(1), #旋轉 90 度
            iaa.Rot90(3), #旋轉 270 度
            iaa.AddToBrightness((-50, 50)),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.Resize({"height": 300, "width": 300})
            ])
        # 得到當前運行的目錄和目錄當中的文件，其中sub_folders可以為空
        for root, sub_folders, files in os.walk(self.Annotation_Input_dir):
            # 遍歷沒一張圖片
            for name in files:
                if (format_mode=="xml"):
                    bndbox = self.Read_xml_annotation(root, name)
                elif (format_mode=="txt"):
                    bndbox = self.Read_txt_annotation(root, name,img_file_extension)
                for epoch in range(aug_multi):
                    aug_det = aug.to_deterministic()  # 保持座標和圖像同步改變，而不是隨機
                    # 讀取圖片
                    img = Image.open(os.path.join(self.ChangeRoot(self.Input_dir,root), name[:-4] + img_file_extension))
                    img = np.array(img)
                    # bndbox 座標增強，依次處理所有的bbox
                    for i in range(len(bndbox)):
                        bbs = ia.BoundingBoxesOnImage([
                            ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                        ], shape=img.shape)
                        bbs_aug = aug_det.augment_bounding_boxes([bbs])[0]
                        boxes_img_aug_list.append(bbs_aug)
                        # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                        new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                                int(bbs_aug.bounding_boxes[0].y1),
                                                int(bbs_aug.bounding_boxes[0].x2),
                                                int(bbs_aug.bounding_boxes[0].y2)])
                    # 存儲變化後的圖片
                    image_aug = aug_det.augment_images([img])[0]
                    #path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    super().CheckDir(super().ChangeRoot(self.Output_dir,root))
                    path = os.path.join(super().ChangeRoot(self.Output_dir,root), str(name[:-4]) + "_" + str(epoch) + img_file_extension)
                    # image_auged = bbs.draw_on_image(image_aug, thickness=0)
                    Image.fromarray(image_aug).save(path)
                    # 存儲變化後的XML
                    if (format_mode=="xml"):
                        self.Change_xml_list_annotation(root, name[:-4], image_aug.shape[1], image_aug.shape[0], new_bndbox_list,super().ChangeRoot(self.Output_dir,root),epoch)
                    elif (format_mode=="txt"):
                        self.Change_txt_list_annotation(root, name[:-4], image_aug.shape[1], image_aug.shape[0], new_bndbox_list,super().ChangeRoot(self.Output_dir,root),epoch,img_file_extension)
                    #print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    new_bndbox_list = []
   
    def Crop(self, aug_multi, img_file_extension, imgSize, format_mode="xml"):
        super().CheckDir(self.Output_dir)
        boxes_img_aug_list = []
        new_bndbox_list = []
        W, H = imgSize
        x1, y1 = 300, 105    # 左上角
        x2, y2 = 1499, 931# 右下角
        # 计算四边裁切的像素数
        top = y1
        right = W - x2
        bottom = H - y2
        left = x1
        aug = iaa.Crop(px=(top, right, bottom, left))
        # 得到當前運行的目錄和目錄當中的文件，其中sub_folders可以為空
        for root, sub_folders, files in os.walk(self.Annotation_Input_dir):
            # 遍歷沒一張圖片
            for name in files:
                if (format_mode=="xml"):
                    bndbox = self.Read_xml_annotation(root, name)
                elif (format_mode=="txt"):
                    bndbox = self.Read_txt_annotation(root, name,img_file_extension)
                for epoch in range(aug_multi):
                    aug_det = aug.to_deterministic()  # 保持座標和圖像同步改變，而不是隨機
                    # 讀取圖片
                    img = Image.open(os.path.join(self.ChangeRoot(self.Input_dir,root), name[:-4] + img_file_extension))
                    img = np.array(img)
                    # bndbox 座標增強，依次處理所有的bbox
                    for i in range(len(bndbox)):
                        bbs = ia.BoundingBoxesOnImage([
                            ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                        ], shape=img.shape)
                        bbs_aug = aug_det.augment_bounding_boxes([bbs])[0]
                        boxes_img_aug_list.append(bbs_aug)
                        # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                        new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                                int(bbs_aug.bounding_boxes[0].y1),
                                                int(bbs_aug.bounding_boxes[0].x2),
                                                int(bbs_aug.bounding_boxes[0].y2)])
                    # 存儲變化後的圖片
                    image_aug = aug_det.augment_images([img])[0]
                    #path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    super().CheckDir(super().ChangeRoot(self.Output_dir,root))
                    path = os.path.join(super().ChangeRoot(self.Output_dir,root), str(name[:-4]) + "_" + str(epoch) + img_file_extension)
                    # image_auged = bbs.draw_on_image(image_aug, thickness=0)
                    Image.fromarray(image_aug).save(path)
                    # 存儲變化後的XML
                    if (format_mode=="xml"):
                        self.Change_xml_list_annotation(root, name[:-4], image_aug.shape[1], image_aug.shape[0], new_bndbox_list,super().ChangeRoot(self.Output_dir,root),epoch)
                    elif (format_mode=="txt"):
                        self.Change_txt_list_annotation(root, name[:-4], image_aug.shape[1], image_aug.shape[0], new_bndbox_list,super().ChangeRoot(self.Output_dir,root),epoch,img_file_extension)
                    #print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    new_bndbox_list = []
    
    def Resize(self, aug_multi, img_file_extension, targetSize, format_mode="xml"):
        super().CheckDir(self.Output_dir)
        boxes_img_aug_list = []
        new_bndbox_list = []
        W, H = targetSize
        aug = iaa.Resize((W, H))
        # 得到當前運行的目錄和目錄當中的文件，其中sub_folders可以為空
        for root, sub_folders, files in os.walk(self.Annotation_Input_dir):
            # 遍歷沒一張圖片
            for name in files:
                if (format_mode=="xml"):
                    bndbox = self.Read_xml_annotation(root, name)
                elif (format_mode=="txt"):
                    bndbox = self.Read_txt_annotation(root, name,img_file_extension)
                for epoch in range(aug_multi):
                    aug_det = aug.to_deterministic()  # 保持座標和圖像同步改變，而不是隨機
                    # 讀取圖片
                    img = Image.open(os.path.join(self.ChangeRoot(self.Input_dir,root), name[:-4] + img_file_extension))
                    img = np.array(img)
                    # bndbox 座標增強，依次處理所有的bbox
                    for i in range(len(bndbox)):
                        bbs = ia.BoundingBoxesOnImage([
                            ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                        ], shape=img.shape)
                        bbs_aug = aug_det.augment_bounding_boxes([bbs])[0]
                        boxes_img_aug_list.append(bbs_aug)
                        # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                        new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                                int(bbs_aug.bounding_boxes[0].y1),
                                                int(bbs_aug.bounding_boxes[0].x2),
                                                int(bbs_aug.bounding_boxes[0].y2)])
                    # 存儲變化後的圖片
                    image_aug = aug_det.augment_images([img])[0]
                    #path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    super().CheckDir(super().ChangeRoot(self.Output_dir,root))
                    path = os.path.join(super().ChangeRoot(self.Output_dir,root), str(name[:-4]) + "_" + str(epoch) + img_file_extension)
                    # image_auged = bbs.draw_on_image(image_aug, thickness=0)
                    Image.fromarray(image_aug).save(path)
                    # 存儲變化後的XML
                    if (format_mode=="xml"):
                        self.Change_xml_list_annotation(root, name[:-4], image_aug.shape[1], image_aug.shape[0], new_bndbox_list,super().ChangeRoot(self.Output_dir,root),epoch)
                    elif (format_mode=="txt"):
                        self.Change_txt_list_annotation(root, name[:-4], image_aug.shape[1], image_aug.shape[0], new_bndbox_list,super().ChangeRoot(self.Output_dir,root),epoch,img_file_extension)
                    #print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    new_bndbox_list = []



if __name__ == '__main__':
    # CNN=CNN()
    # CNN.Augment(20, ".bmp", mode="Sequential")
    OD=OD()
    # OD.Augment(20,".bmp",mode="Sequential",format_mode="txt")
    # OD.Crop(1,".jpg", (1920, 1080), format_mode="txt")
    OD.Resize(1,".jpg", (300, 300), format_mode="txt")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
