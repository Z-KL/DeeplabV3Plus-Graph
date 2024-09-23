import os.path
import subprocess
from detect.test import *
from detect.train import *
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing
import xlwt
from temporal_analysis import *

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
backbone = 'resnet50'
pretrained = 'imagenet'
DEVICE = 'cuda'


def augBC(imgs_dir,imgs_aug_dir,sequence,mask_dir,mask_aug_dir,mask):
    contrast_list = [0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
    contrast_type=len(contrast_list)+1
    img_num=sequence
    img_name=str(sequence).zfill(6)+".tif"
    img=cv2.imread(os.path.join(imgs_dir,img_name),-1)
    cv2.imwrite(os.path.join(imgs_aug_dir,str(img_num*contrast_type).zfill(6)+".tif"),img)
    if mask:
        img_mask=cv2.imread(os.path.join(mask_dir,img_name),-1)
        for i in range(contrast_type):
            cv2.imwrite(os.path.join(mask_aug_dir, str(img_num * contrast_type+i).zfill(6) + ".tif"), img_mask)
    img=img.astype(np.float32)
    print("generating contrast augmentation for image:",img_name)
    for j,contrast in enumerate(contrast_list,start=1):
        img_C=np.clip(img*contrast,0,255)
        img_C=img_C.astype(np.uint8)
        cv2.imwrite(os.path.join(imgs_aug_dir,str(img_num*contrast_type+j).zfill(6)+".tif"),img_C)

def augmentationWithPool(imgs_dir,imgs_aug_dir,mask_dir=None,mask_aug_dir=None,mask=False):
    createFolder(imgs_aug_dir,clean=True)
    if mask:
        createFolder(mask_aug_dir,clean=True)
    img_list=os.listdir(imgs_dir)
    img_list.sort()
    print(img_list)
    p=multiprocessing.Pool()
    for i in range(len(img_list)):
        p.apply_async(augBC,args=(imgs_dir,imgs_aug_dir,i,mask_dir,mask_aug_dir,mask,))
    p.close()
    p.join()

def test_and_process(img_path,img_aug_path,ckpt_path,current_path,next_path):
    predict_path=os.path.join(current_path,"predict")
    predict_result_path=os.path.join(os.path.join(current_path,"predict_result"))
    weighted_sum_path=os.path.join(current_path,"weighted_sum")
    temporal_path=os.path.join(current_path,"temporal")
    temporal_mask_path=os.path.join(temporal_path,"mask")
    next_mask_path=os.path.join(next_path,"mask")
    mask_path=os.path.join(current_path,"mask")
    createFolder(predict_path, clean=True)
    createFolder(predict_result_path, clean=True)
    createFolder(weighted_sum_path, clean=True)
    createFolder(temporal_path, clean=True)
    createFolder(next_mask_path, clean=True)
    createFolder(temporal_mask_path,clean=True)
    ckpt_list = os.listdir(ckpt_path)
    ckpt_list.sort()
    final_ckpt = os.path.join(ckpt_path, ckpt_list[-1])

    #predict
    print("predicting images with ckpt:", final_ckpt)
    test(img_aug_path, predict_path, final_ckpt)
    predict_list = os.listdir(predict_path)
    predict_list = [name for name in predict_list if ".tif" in name]
    predict_list.sort()
    predict_imgs = []
    print("\tprocessing predicted mask in:", predict_path)
    for name in predict_list:
        result = cv2.imread(os.path.join(predict_path, name), -1)
        img = result.astype(np.float32)
        predict_imgs.append(img)
        ret, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(predict_result_path, name), result)

    print("\tstart weighted sum in :",weighted_sum_path)
    #加权和并结合初始标签
    contrast_type = int(len(predict_imgs) / 92)
    for i in range(0, len(predict_imgs), contrast_type):
        mask_combine = np.zeros((predict_imgs[i].shape), dtype=np.float32)
        for j in range(i, i + contrast_type):
            mask_combine += predict_imgs[j] * 0.125
        weighted_sum = mask_combine.astype(np.uint8)
        cv2.imwrite(os.path.join(weighted_sum_path, str(i // contrast_type).zfill(6) + ".tif"), weighted_sum)

        # 加上初始标签
        mask_old = cv2.imread(os.path.join(mask_path, str(i).zfill(6) + ".tif"), -1)
        ret,mask_combine_binary=cv2.threshold(weighted_sum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask_final = mask_old + mask_combine_binary
        mask_final = np.clip(mask_final,0,255)
        mask_final=mask_final.astype(np.uint8)
        mask_final = useAreaFilter(mask_final, 40)
        cv2.imwrite(os.path.join(temporal_mask_path, str(i // contrast_type).zfill(6) + ".tif"), mask_final)

    #追踪分析
    print("\tstart analyzing by temporal consistency")
    temporal_track_path=os.path.join(temporal_path,"track")
    createFolder(temporal_track_path,clean=True)
    #track_all_periods(img_aug_path,temporal_mask_path,temporal_track_path)
    track_single_period(img_path,temporal_track_path,temporal_mask_path,remove_edge=False)
    #reduceFP_with_Pool(img_path,temporal_track_path)
    reduceFP_use_track(img_path,temporal_track_path)
    print("\tgenerating new mask")

    remove_FP_path=os.path.join(temporal_track_path,"remove_FP")
    length=len(os.listdir(next_mask_path))
    mask_without_FP_list=os.listdir(remove_FP_path)
    mask_without_FP_list.sort()
    for j in range(len(mask_without_FP_list)):
        mask=cv2.imread(os.path.join(remove_FP_path,mask_without_FP_list[j]),-1)
        for k in range(contrast_type):
            cv2.imwrite(os.path.join(next_mask_path,str(length+j*contrast_type+k).zfill(6)+".tif"),mask)


def trainWithIteration(img_path,img_aug_path,iteration_path,keep_training,iteration_times=5,batch_size=32,data_shuffle=False):
    x_transforms = transforms.Compose([
        #transforms.Lambda(lambda image: clahe(image)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    # model = Unet(1, 1)
    model = smp.DeepLabV3Plus(backbone, in_channels=1, encoder_weights=pretrained)
    criterion = nn.BCEWithLogitsLoss()  # 包含sigmoid
    optimizer = optim.Adam(model.parameters())
    ckpt_path = os.path.join(iteration_path, "checkpoints")
    createFolder(ckpt_path,clean=True)

    old_loss=999
    for i in range(iteration_times):
        if i==0:
            training=False
        else:
            training=keep_training
        current_folder=os.path.join(iteration_path,"iteration"+str(i+1).zfill(2))
        mask_path=os.path.join(current_folder,"mask")
        print("\nstart iteration {}/{}: {}----{}".format(i + 1,iteration_times, img_aug_path,mask_path))
        data = TrainDataset(img_aug_path, mask_path, x_transforms, y_transforms)
        dataloader = DataLoader(data, batch_size, shuffle=data_shuffle, num_workers=4)
        loss=train_model(model, criterion, optimizer, dataloader, training, ckpt_path, num_epochs=1)

        next_folder=os.path.join(iteration_path,"iteration"+str(i+2).zfill(2))
        createFolder(next_folder)
        test_and_process(img_path,img_aug_path,ckpt_path,current_folder,next_folder)

        '''if loss<old_loss:
            old_loss=loss
        else:
            break'''
def evaluate_DET_SEG_TRA(bright_field,track_result_path,isWin=False):

    res_path=os.path.join(track_result_path,"track_RES")
    bright_field_num=str(bright_field).zfill(2)
    data_folder="EvaluationSoftware/"+bright_field_num
    evaluation_path=os.path.join(data_folder,bright_field_num+"_RES")
    deleteFile(evaluation_path)
    copyFile(res_path, evaluation_path)
    if isWin:
        evaluate_DET_command = "/EvaluationSoftware/Win/DETMeasure.exe" + " " + data_folder + \
                               " " + bright_field_num + " " + "6"
        result, DET = subprocess.getstatusoutput(evaluate_DET_command)
        evaluate_SEG_command = "/EvaluationSoftware/Win/SEGMeasure.exe" + " " + data_folder + \
                               " " + bright_field_num + " " + "6"
        result, SEG = subprocess.getstatusoutput(evaluate_SEG_command)
        evaluate_TRA_command = "/EvaluationSoftware/Win/TRAMeasure.exe" + " " + data_folder + \
                               " " + bright_field_num + " " + "6"
        result, TRA = subprocess.getstatusoutput(evaluate_TRA_command)
    else:
        evaluate_DET_command = "EvaluationSoftware/Linux/DETMeasure"+" " + data_folder +\
                               " "+bright_field_num+" "+"6"
        result, DET = subprocess.getstatusoutput(evaluate_DET_command)
        evaluate_SEG_command = "EvaluationSoftware/Linux/SEGMeasure" + " " + data_folder + \
                               " " + bright_field_num + " " + "6"
        result, SEG = subprocess.getstatusoutput(evaluate_SEG_command)
        evaluate_TRA_command = "EvaluationSoftware/Linux/TRAMeasure" +" "+ data_folder  +\
                               " " + bright_field_num + " " + "6"
        result, TRA = subprocess.getstatusoutput(evaluate_TRA_command)
    DET=float(DET.replace("DET measure: ",""))
    SEG=float(SEG.replace("SEG measure: ",""))
    TRA=float(TRA.replace("TRA measure: ",""))
    print(DET,SEG,TRA)
    deleteFile(res_path)
    copyFile(evaluation_path,res_path)
    det_log_path=os.path.join(res_path,"DET_log.txt")
    with open(det_log_path, "r") as f:
        data = f.readlines()
    for index, line in enumerate(data):
        if "Splitting Operations" in line:
            num1 = index
        if "False Negative" in line:
            num2 = index
        if "False Positive" in line:
            num3 = index
    split = num2 - num1 - 1
    FN = num3 - num2 - 1
    FP = len(data) - num3 - 1

    track_list=os.listdir(res_path)
    track_list=[name for name in track_list if ".tif" in name]
    positive=0
    for name in track_list:
        mask=cv2.imread(os.path.join(res_path,name),-1)
        positive+=(len(np.unique(mask))-1)
    TP = positive - FP
    precision = TP / positive
    recall = TP / (TP + FN)
    F_measure = 2 * precision * recall / (precision + recall)
    FN_per_img = FN / (len(track_list))
    FP_per_img = FP / (len(track_list))

    '''GT_path=os.path.join(data_folder,bright_field_num+"_GT/TRA")
    GT_list=os.listdir(GT_path)
    cell_num=0
    GT_list=[name for name in GT_list if ".tif" in name]
    for name in GT_list:
        GT=cv2.imread(os.path.join(GT_path,name),-1)
        cell_num+=(len(np.unique(GT))-1)
    cell_per_img=int(cell_num/len(GT_list))'''

    print("======FN:{}  len(track_list):{}  FN_per_img:{}======".format(FN,len(track_list),FN_per_img))

    '''FN_per_img=str(FN_per_img)+"/"+str(cell_per_img)
    FP_per_img=str(FP_per_img)+"/"+str(cell_per_img)'''
    return DET,SEG,TRA,precision,recall,F_measure,FN_per_img,FP_per_img

def record_performance(excel_path,DET,SEG,TRA,precision,recall,F_measure,FN_per_img,FP_per_img):
    print("DET:",DET)
    print("SEG:",SEG)
    print("TRA:",TRA)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-measure:",F_measure)
    book=xlwt.Workbook()
    DET_sheet=book.add_sheet('DET')
    SEG_sheet=book.add_sheet('SEG')
    TRA_sheet=book.add_sheet('TRA')
    pre_sheet=book.add_sheet('Precision')
    recall_sheet=book.add_sheet('Recall')
    f_sheet=book.add_sheet('F_measure')
    FN_sheet=book.add_sheet('FN_per_img')
    FP_sheet=book.add_sheet('FP_per_img')
    iteration_times=int(len(DET))
    for i in range(iteration_times):
        DET_sheet.write(0,i+1,"iteration"+str(i+1).zfill(2))
        SEG_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        TRA_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        pre_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        recall_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        f_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        FN_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))
        FP_sheet.write(0, i + 1, "iteration" + str(i + 1).zfill(2))

        DET_sheet.write(1,i+1,DET[i])
        SEG_sheet.write(1,i+1,SEG[i])
        TRA_sheet.write(1,i+1,TRA[i])
        pre_sheet.write(1, i + 1, precision[i])
        recall_sheet.write(1, i + 1, recall[i])
        f_sheet.write(1, i + 1, F_measure[i])
        FN_sheet.write(1, i + 1, FN_per_img[i])
        FP_sheet.write(1, i + 1, FP_per_img[i])
    for j in range(1):
        DET_sheet.write(j+1,0,"period"+str(j+1).zfill(2))
        SEG_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        TRA_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        pre_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        recall_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        f_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        FN_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))
        FP_sheet.write(j + 1, 0, "period" + str(j + 1).zfill(2))

    book.save(excel_path)
    print("\t\t\t performance has been recorded in ",excel_path)

def verify_and_evaluate(verify_bright_field_num,ckpt_path,result_dir):
    img_path = os.path.join(str(verify_bright_field_num) + "-GT", "imgs")
    ckpts = os.listdir(ckpt_path)
    ckpts.sort()
    img_path_list=[]
    track_path_list=[]
    result_path_list=[]
    predict_path_list=[]
    for i in range(len(ckpts)):
        result_path=os.path.join(result_dir,ckpts[i].replace(".pth","")+"_result")
        predict_path = os.path.join(result_path, "predict")
        createFolder(result_path)
        createFolder(predict_path)

        ckpt=os.path.join(ckpt_path, ckpts[i])
        print("verify using {}, saved in {}.".format(ckpt, predict_path))
        test(img_path, predict_path, ckpt)
        track_path = os.path.join(result_path, "track")
        createFolder(track_path)

        track_path_list.append(track_path)
        img_path_list.append(img_path)
        predict_path_list.append(predict_path)
        result_path_list.append(result_path)
    with Pool() as p:
        p.map(track_single_period,img_path_list,track_path_list,predict_path_list)

    DET_result = []
    SEG_result = []
    TRA_result = []
    precision = []
    recall = []
    F_measure = []
    FN_per_img = []
    FP_per_img = []
    for folder in result_path_list:
        track_path=os.path.join(folder,"track")
        DET, SEG,TRA, pre, cal, fm, fnpi, fppi = evaluate_DET_SEG_TRA(bright_field=verify_bright_field_num, track_result_path=track_path)
        DET_result.append(DET)
        SEG_result.append(SEG)
        TRA_result.append(TRA)
        precision.append(pre)
        recall.append(cal)
        F_measure.append(fm)
        FN_per_img.append(fnpi)
        FP_per_img.append(fppi)
    result_excel = os.path.join(result_dir, "verify_result.xls")
    record_performance(result_excel, DET_result,SEG_result, TRA_result, precision, recall, F_measure, FN_per_img, FP_per_img)

def track_with_pool(img_path,folder_path):
    print(folder_path)
    mask_path = os.path.join(folder_path, "mask")
    print(mask_path)
    track_path = os.path.join(folder_path, "track")
    createFolder(track_path,clean=True)
    track_single_period(img_path,track_path,mask_path)

def iteration_and_evaluate(bright_field_num,iteration_times,keep_training,verify_BF_num,batch_size=32,shuffle=False):
    img_path=str(bright_field_num)+"-GT/imgs"
    mask_path=str(bright_field_num)+"-GT/mask"

    iteration_path=str(bright_field_num)+"-Iteration"
    img_aug_path = os.path.join(iteration_path, "imgs_aug")
    first_iteration_path = os.path.join(iteration_path, "iteration01")
    mask_aug_path = os.path.join(first_iteration_path, "mask")

    createFolder(iteration_path,clean=True)
    createFolder(img_aug_path)
    createFolder(first_iteration_path)
    createFolder(mask_aug_path)

    augmentationWithPool(img_path, img_aug_path, mask_path, mask_aug_path, mask=True)
    trainWithIteration(img_path,img_aug_path, iteration_path, keep_training, iteration_times,batch_size,shuffle)

    iteration_folder_list = os.listdir(iteration_path)
    iteration_folder_list=[name for name in iteration_folder_list if "iteration" in name]
    iteration_folder_list.sort()
    print(iteration_folder_list)


    img_path_list = []
    for i in range(len(iteration_folder_list)):
        img_path_list.append(img_path)
    folder_path_list=[os.path.join(iteration_path,name) for name in iteration_folder_list]
    with Pool() as p:
        p.map(track_with_pool,img_path_list,folder_path_list)

    DET_result = []
    SEG_result = []
    TRA_result = []
    precision=[]
    recall=[]
    F_measure=[]
    FN_per_img=[]
    FP_per_img=[]
    for folder in iteration_folder_list:
        folder_path = os.path.join(iteration_path, folder)
        track_path = os.path.join(folder_path, "track")
        DET, SEG,TRA,pre,cal,fm,fnpi,fppi = evaluate_DET_SEG_TRA(bright_field=bright_field_num,track_result_path=track_path)
        DET_result.append(DET)
        SEG_result.append(SEG)
        TRA_result.append(TRA)
        precision.append(pre)
        recall.append(cal)
        F_measure.append(fm)
        FN_per_img.append(fnpi)
        FP_per_img.append(fppi)
    result_excel = os.path.join(iteration_path, "evaluation_result.xls")
    record_performance(result_excel, DET_result,SEG_result, TRA_result,precision,recall,F_measure,FN_per_img,FP_per_img)

    print("==========Strating verfify==========")
    verfiy_result_path=os.path.join(iteration_path,"verify")
    createFolder(verfiy_result_path,clean=True)
    checkpoint_path=os.path.join(iteration_path,"checkpoints")
    verify_and_evaluate(verify_BF_num,checkpoint_path,verfiy_result_path)


if __name__=="__main__":
    iteration_times=2
    keep_training=True
    shuffle=True
    batch_size=8
    train_field=2
    test_field=1
    iteration_and_evaluate(train_field,iteration_times,keep_training,test_field,batch_size,shuffle)
