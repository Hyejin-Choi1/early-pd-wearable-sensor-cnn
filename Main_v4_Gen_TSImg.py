# from data_specific.Proc_TS_Dataset import Get_TS_Datasets, Save_TSImgDatasetInfo, Load_TSImgDatasetInfo
# from data_specific.Proc_Features import Save_Features_FromTSDatasets
# from data_specific.Proc_TSImgs import Save_TSImgs_FromTSDatasets, Load_TSImgs_PT
# from LibKIME.LibML import Split_Dataset_PT

import sys, os
from LibKIME.TSImg import TSImg
from LibKIME.LibGeneral import GetNowString, MakeFolder

##################
### Parameters ###
##################
from data_specific.LibTSDataset import Save_TSImgDatasetInfo
from data_specific.Proc_TSImgs import Save_TSImgs_FromTSDatasets


# csv_data_root_folder = r'D:\ws\TS\박화영_PD\RotPDTSv3(PD)_221121_@'
# tsimg_root_path = r'D:\ws\TS\PD_TSImgs_231020'
#csv_data_root_folder = r'D:\WorkSpace\TS\박화영_PD\RotPDTSv3(PD)_221121_@'
#tsimg_root_path = r'D:\WorkSpace\TS\PD_TSImgs_231020'
#import data_specific.PD_Park.Info as Info
#import data_specific.PD_Park.Task as Task

#csv_data_root_folder = r'D:\ws\TS\김보현_Cognitive\Data\224_2group\CFMF\Faster'
#tsimg_root_path = r'D:\ws\TS\224_2group_CFMF_Faster_TSImgs_231020'
#import data_specific.Cog_Kim.Info as Info
#import data_specific.PD_Park.Task as Task

csv_data_root_folder = r'G:\Toolkit\Backup\6MWT_CNN_data_2024_241011\Image_generating\Linear_filterd\Thora_896'
tsimg_root_path = r'G:\Toolkit\Backup\6MWT_CNN_data_2024_241011\Image_generating\Imaging\Linear\Thora_896'
import data_specific.Walk_6m_Choi.Info as Info
import data_specific.PD_Park.Task as Task



def main():
    # This procedure (Get_TS_Datasets) includes
    # Step1) Load raw data (csv & mat, etc.)
    # Step2) Preprocessing (filtering, gating, length, etc.)
    # Step3) Imbalanced data procedure
    # Step4) Prepare TS_Dataset Structure

    info_ds = {'GeneratedDate': GetNowString(),
               'FSNames': Info.TS_DataSetInfo.FS_Names,
               'Classes': Info.TS_DataSetInfo.Classes,
               'ClasseNames': list(map(str, Info.TS_DataSetInfo.Classes)),
               'RawData_RelPath': Info.TS_DataSetInfo.RawData_RelPath,
               'TSImgs': Info.TS_DataSetInfo.TSImg_Methods,
               'TSImgNames': list(map(str, Info.TS_DataSetInfo.TSImg_Methods)),
               'Imbalanced': "",  # No Imbalance Procedure @ Making TSImages
               }

    ds_TS_raw = Task.Load_TS_Dataset(csv_data_root_folder, info_ds)

    MakeFolder(tsimg_root_path)
    for ds in ds_TS_raw:
        ds_TS_raw[ds].Save_Info(os.path.join(tsimg_root_path, f"RawTS_{ds}.json"))

    ds_TS = ds_TS_raw
    Save_TSImgs_FromTSDatasets(ds_TS, info_ds, tsimg_root_path)

    # Save TSImg Dataset Info
    Save_TSImgDatasetInfo(info_ds, tsimg_root_path)


if __name__ == "__main__":
    main()
