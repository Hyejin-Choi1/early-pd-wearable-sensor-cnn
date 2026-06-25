import os, sys
import matplotlib.pyplot as plt
from itertools import chain, product
import numpy as np
import torch
import torchvision

from LibKIME.LibGeneral import (MakeFolder,
                                StartTimer,
                                StopTimer,
                                Save_dict2json,
                                LOGGER,
                                GetNowString)
from LibKIME.LibML_Exp import Exp_Detail, CV_Loop


##################
### Working Folder ###
##################
#csv_data_root_folder = r'C:\Users\chjin\OneDrive\바탕 화면\Phase1_12_24\Phase1\Larm'
#tsimg_root_path = r'D:\ws\TS\PD_TSImgs_231020'
#dl_result_root_path = r'C:\Users\chjin\OneDrive\바탕 화면\Imaging_Phase1_12_24\Phase1'
#import data_specific.PD_Park.Info as Info
#import data_specific.PD_Park.Task as Task

# csv_data_root_folder = r'D:\ws\TS\김보현_Cognitive\Data\224_2group\CFMF\Faster'
# tsimg_root_path = r'D:\ws\TS\224_2group_CFMF_Faster_TSImgs_231020'
# dl_result_root_path = r'D:\ws\TS\224_2group_CFMF_Faster_TSImgs_Result_231020'
# import data_specific.Cog_Kim.Info as Info
# import data_specific.PD_Park.Task as Task

csv_data_root_folder = r'C:\Users\User\Desktop\CHJ\6MWT_CNN_data_2024_241016\Raw_linear_EarlyPDs\Thora_896'
tsimg_root_path = r'C:\Users\User\Desktop\CHJ\6MWT_CNN_data_2024_241016\Imaging_linear_EarlyPDs\Thora_896'
dl_result_root_path = r'C:\Users\User\Desktop\CHJ\6MWT_CNN_data_2024_241016\Results_linear_EarlyPDs\Thora_896\CNN'
import data_specific.Walk_6m_Choi.Info as Info
import data_specific.PD_Park.Task as Task



def main():
    # This procedure (Experiment) includes
    # Step1) Load TS_Dataset data
    # Step2) Train & Test (5-CV)

    # Experiment Parameters
    tsimg_methods = Info.TSImg_Exp_Info.exp_tsimg
    info_fs = Info.TSImg_Exp_Info.exp_fsname
    include_classes = Info.TSImg_Exp_Info.exp_include_classes
    split_ratio = Info.TSImg_Exp_Info.exp_split_ratio
    cv = Info.TSImg_Exp_Info.exp_cv
    num_classes = Info.TSImg_Exp_Info.exp_num_classes
    mdlinfos = Info.Models


    path = os.path.join(dl_result_root_path,
                        f"TSImg_Exp_Info_{GetNowString(bFileFormat=True)}.json")
    Save_dict2json(path, dict(Info.TSImg_Exp_Info.__dict__))

    start = StartTimer()
    Results = {}
    Loop_for_ExcludeCalculating = []

    # Make Result Folders
    # &
    # Load Previously Saved Data to Skip
    for tsimg, fsname in product(tsimg_methods, info_fs):
        ColNames = info_fs[fsname][1].get_ColNames()  # 0: All, 1: Select
        for col in ColNames:
            result_path = os.path.join(dl_result_root_path, str(tsimg), fsname, col)
            MakeFolder(result_path)
            exp_key = (tsimg, fsname, col)

            Inter_Result_fn = os.path.join(result_path, f"Intermediate_Result.pt")
            if os.path.isfile(Inter_Result_fn):
                try:
                    Results[exp_key] = torch.load(Inter_Result_fn)
                    Loop_for_ExcludeCalculating.append(exp_key)
                except FileNotFoundError as err:
                    LOGGER.error("There is no saved files. Continue the training.")
                except Exception as err:
                    LOGGER.error(f"Loading Intermediate Result - {err}")


    # Do Experiment (Train, Test, Analysis)
    for tsimg, fsname in product(tsimg_methods, info_fs):
        ColNames = info_fs[fsname][1].get_ColNames()  # 0: All, 1: Select
        for col in ColNames:
            result_path = os.path.join(dl_result_root_path, str(tsimg), fsname, col)
            exp_key = (tsimg, fsname, col)
            if (exp_key in Loop_for_ExcludeCalculating): continue  # Skip this experiment
            LOGGER.info(f"{tsimg}-{fsname}-{col}")


            #####################################################################
            # Load Pytorch Dataset & CV by Subject
            #####################################################################
            # Load Pytorch Dataset
            ds_pt = Task.Load_Dataset(tsimg_root_path, tsimg, fsname, col, include_classes)
            assert(num_classes == len(ds_pt.classes))

            # CV by Subject
            df, out_cv_ind_bysubject = Task.Split_Dataset_bySubject(ds_pt, split_ratio, cv)
            title=f'{str(tsimg)}_{fsname}_{col}'
            Save_dict2json(os.path.join(result_path, f"df_{title}.json"), df.to_dict())
            Save_dict2json(os.path.join(result_path, f"CV_bySubject_{title}.json"), out_cv_ind_bysubject)

            # Sample Level Index from CV Subject
            out_cv_ind_sample = Task.Get_Samples_Of_Subject(df, out_cv_ind_bysubject)
            title=f'{str(tsimg)}_{fsname}_{col}'
            Save_dict2json(os.path.join(result_path, f"CV_SamplesbySub_{title}.json"), out_cv_ind_sample)

            # Imbalanced Data Handle After 5 Fold CV (by Subject)
            out_cv_ind_imb = Task.Imbalance_Samples(df, out_cv_ind_sample, Info.TSImg_Exp_Info.exp_imbalanced)
            title=f'{str(tsimg)}_{fsname}_{col}'
            Save_dict2json(os.path.join(result_path, f"CV_ImbalancedSambySub_{title}.json"), out_cv_ind_imb)

            exp = Exp_Detail(
                (tsimg, fsname, col),
                mdlinfos,
                ds_pt,
                out_cv_ind_imb["Train_Index"],
                out_cv_ind_imb["Val_Index"],
                out_cv_ind_imb["Test_Index"])

            for mdl in exp.models:
                for cv_i in range(cv):
                    mdl.CVs[cv_i] = CV_Loop(mdl.MdlInfo, mdl.CVs[cv_i])
                    mdl.CVs[cv_i].SavePlot(result_path, title=f"{mdl.MdlInfo.model_name}-CV{cv_i}")
                mdl.SavePlot(result_path, title=f"{mdl.MdlInfo.model_name}")

            # Save Intermediate Results after all exp loops
            Results[exp_key] = exp
            torch.save(Results[exp_key],
                       os.path.join(result_path, f"Intermediate_Result.pt"))

            Results[exp_key].SaveResults(result_path,
                                         title=f'{str(tsimg)}_{fsname}_{col}')
            ## Plot Acc - for ALL Exp
            Results[exp_key].SavePlot(result_path,
                                      title=f'{str(tsimg)}_{fsname}_{col}')

    StopTimer(start)



if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  #Window 환경에서의 RuntimeError 방지 (DataLoader의 num_workers 관련)
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # Plot Init
    plt.close('all')
    # plt.rcParams["font.family"] = "Times New Roman"

    main()
