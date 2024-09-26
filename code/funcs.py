from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np
import moviepy.editor as moviepy
import os
import shutil

from typing import List

# import yaml
# from moviepy.editor import VideoFileClip
# from PIL import Image
# import json

from lightning_pose.utils.predictions import predict_dataset
from lightning_pose.utils.scripts import (
    export_predictions_and_labeled_video,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
    compute_metrics,
)

def get_keypoint_names(csv_file: str, header_rows: List[int]) -> List[str]:
    """ get the bodypart names given the .csv file
    
    Args:
        csv_file: the prediction/evaluation .csv file
        header_rows: multiindex header of the .csv file

    Returns:
        list of bodypart names
    """
    if os.path.exists(csv_file):
        csv_data = pd.read_csv(csv_file, header=header_rows)
        # collect marker names from multiindex header
        if header_rows == [1, 2] or header_rows == [0, 1]:
            keypoint_names = [b[0] for b in csv_data.columns if b[1] == "x"]
        elif header_rows == [0, 1, 2]:
            keypoint_names = [b[1] for b in csv_data.columns if b[2] == "x"]
    else:
        # keypoint_names = ["bp_%i" % n for n in range(cfg.data.num_targets // 2)]
        keypoint_names = []
        print("keypoint_names do not exist!!!")
    return keypoint_names


def get_videos_in_dir(video_dir: str, return_mp4_only: bool=False) -> List[str]:
    """Get all video files in directory given allowed formats """
    # gather videos to process
    print(f"video_dir: {video_dir}")
    assert os.path.isdir(video_dir)

    allowed_formats = (".mp4", ".avi", ".mov")
    if return_mp4_only == True:
        allowed_formats = ".mp4"
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(allowed_formats)]

    if len(video_files) == 0:
        raise IOError("Did not find any valid video files in %s" % video_dir)
    return video_files


def convert_header_to_dlc(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """ Convert the header of dataframe to DLC format"""
    df_arry = df.to_numpy()
    # keypoint_names = [b[0] for b in df.columns if b[1] == "x"]
    keypoint_names = df.columns.get_level_values("bodyparts").unique()
    print(f"The number of keypoint: {len(keypoint_names)}" )
    # model_name = 'DLC_resnet50'
    pdindex = make_labels_dlc_index(model_name, keypoint_names)
    df_dlc_index = pd.DataFrame(df_arry, columns=pdindex, index = df.index)

    return df_dlc_index


def make_labels_dlc_index(model_name: str, keypoint_names: List[str]) -> pd.DataFrame:
    """ Create the MultiIndex with DLC format """
    # xyl_labels = ["x", "y", "likelihood"] # for prediction
    xyl_labels = ["x", "y"] # for ground truth
    pdindex = pd.MultiIndex.from_product(
        [["%s_tracker" % model_name], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    
    return pdindex


def mask_df(file: str, header_rows: List[int], bodyparts: List[str]) -> pd.DataFrame:
    """ Select the columns in bodyparts, return the masked dataframe """
    df = pd.read_csv(file, header=header_rows, index_col=0) 
    mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
    df_masked = df.loc[:, mask]
    
    return df_masked


def closest_multiple128(K: int) -> int:
    """ Find the nearest multiple of 128 to K.
    LP requires the dimension of image dimension is a multiple of 128 to accelerate training.
    """
    if K <= 128:
        return 128
    elif K >= 1280:
        return 1280
    else:
        return int(round(K/128))*128
    
def dlc2lp(
    dlc_dir: str, 
    lp_dir: str, 
    model_name: str, 
    save_lp_csv: str, 
    videos_picked: List[str]) -> pd.DataFrame:
    '''Converting DLC project located at dlc_dir to LP project located at lp_dir
    Call dlc2lp() to generate the LP dataset with the following directory struture
     /path/to/LP_project/
       ├── <LABELED_DATA_DIR>/
       ├── <VIDEO_DIR>/
       └── <YOUR_LABELED_FRAMES>.csv
       
    Args:
        dlc_dir: path to DLC project
        lp_dir: path to save LP project
        model_name: LP model name
        save_lp_csv: path to save the LP labels .csv file
        videos_picked: videos of interest used for generating the LP labels .csv file
        
    Return:
        pd.DataFrame: concatenated labels
    '''
    print(f"\nConverting DLC project located at {dlc_dir} to LP project located at {lp_dir}")
    
    # check provided DLC path exists
    if not os.path.exists(dlc_dir):
        raise NotADirectoryError(f"did not find the directory {dlc_dir}")

    # check paths are not the same
    if dlc_dir == lp_dir:
        raise NameError(f"dlc_dir and lp_dir cannot be the same")
  
    # check videos_picked
    if len(videos_picked) == 0:
        raise ValueError(f"videos_picked cannot be null")
        
    # find all labeled data in DLC project
    dirs = os.listdir(os.path.join(dlc_dir, "labeled-data"))
    dirs.sort()
    dfs = []
    
    #-------------------------------------    
    # Step 1: get the DLC labels in videos_picked and convert to LP format 
    #-------------------------------------    
    print("Start generating <YOUR_LABELED_FRAMES>.csv!")
    # for curr_video in dirs[:num_videos]:
    for curr_video in videos_picked:
        print("----", curr_video, "----",)
        try:
            # assume dlc format
            header_rows = [0, 1, 2]
            
            # read the annotation file of current video 
            csv_file = glob(os.path.join(dlc_dir, "labeled-data", curr_video, "CollectedData*.csv"))[0]
            print(f"csv_file:{csv_file}")
            df_tmp = pd.read_csv(csv_file, header=header_rows, index_col=0)
            
            # convert the .DLC csv_file to LP format
            if len(df_tmp.index.unique()) != df_tmp.shape[0]:
                # print("new DLC labeling scheme that splits video/image in different cells!!")
                # new DLC labeling scheme that splits video/image in different cells
                vids = df_tmp.loc[
                       :, ("Unnamed: 1_level_0", "Unnamed: 1_level_1", "Unnamed: 1_level_2")]
                imgs = df_tmp.loc[
                       :, ("Unnamed: 2_level_0", "Unnamed: 2_level_1", "Unnamed: 2_level_2")]
                # new_col = [f"labeled-data/{v}/{i}" for v, i in zip(vids, imgs)]
                
                # use the acutual video name
                new_col = [f"labeled-data/{curr_video}/{i}" for i in imgs]

                df_tmp1 = df_tmp.drop(
                    ("Unnamed: 1_level_0", "Unnamed: 1_level_1", "Unnamed: 1_level_2"), axis=1,
                )
                df_tmp2 = df_tmp1.drop(
                    ("Unnamed: 2_level_0", "Unnamed: 2_level_1", "Unnamed: 2_level_2"), axis=1,
                )
                df_tmp2.index = new_col
                df_tmp = df_tmp2

                # make the MultiIndex consistent through differet sessions (videos may be annotated by different scorer)
                df_tmp = convert_header_to_dlc(df_tmp, model_name)
                dfs.append(df_tmp)
       
        except IndexError:
            print(f"Could not find labels for {curr_video}; skipping") 
            
    #-----------------------------------------------------------------         
    # Step 2: concatenate the annotation .csv files of videos_picked to
    # <YOUR_LABELED_FRAMES>.csv
    #------------------------------------- ----------------------------   
    df_data = pd.concat(dfs)
    # save concatenated labels
    df_data.to_csv(save_lp_csv)
    print("Finish generating <YOUR_LABELED_FRAMES>.csv!")
    print("-----"*15)
    print()
    
    #-----------------------------------------------------------------         
    # Step 3: copying or generating mp4 video to lp_dir
    # All unlabeled videos must be placed in a single directory. 
    # <VIDEO_DIR>/
    #-----------------------------------------------------------------    
    os.makedirs(lp_dir, exist_ok=True)
    print("Start generating <VIDEO_DIR>/!")
    for curr_video in videos_picked:
        video_file = glob(os.path.join(dlc_dir, "videos", f"{curr_video}.*"))[0]
        print(f"Working on: {video_file}")

        lp_video_dir = os.path.join(lp_dir, 'videos/')
        Path(lp_video_dir).mkdir(parents=True, exist_ok=True)
        
        # convert avi. videos to mp4 format since LP only accepts mp4
        if video_file.endswith(".avi"):
            # save mp4 to lp_video_dir
            inputfile  = video_file[:-4] + '.avi'
            outputfile = video_file[:-4] + '.mp4'
            outputfile = os.path.join(lp_video_dir, outputfile.split('/')[-1])

            if not os.path.exists(outputfile):
                print("Converting avi video files to be mp4 format!")
                print("outputfile:", outputfile) 
                clip = moviepy.VideoFileClip(inputfile)
                clip.write_videofile(outputfile)
        else:
            # Optional: copy video over
            # if the behavior videos are very large, it may take long time
            outputfile = os.path.join(lp_video_dir, video_file.split('/')[-1])     
            if not os.path.exists(outputfile):
                print("copying video files")
                shutil.copyfile(video_file, outputfile)
    print("Finish generating <VIDEO_DIR>/!")
    print("-----"*15)
    print()
    
    #-----------------------------------------------------------------         
    # Step 4: copying frames to lp_dir
    # <LABELED_DATA_DIR>/
    #----------------------------------------------------------------- 
    print("Start generating <LABELED_DATA_DIR>/!")
    for curr_video in videos_picked:  
        # copy frames over
        src = os.path.join(dlc_dir, "labeled-data", curr_video)
        dst = os.path.join(lp_dir, "labeled-data", curr_video)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
    print("Finish generating <LABELED_DATA_DIR>/!")
    print("-----"*15)
    print()
    
    #-----------------------------------------------------------------     
    # Step 5: check if the labeled frames in <YOUR_LABELED_FRAMES>.csv exist
    # make sure the first column of <YOUR_LABELED_FRAMES>.csv matchs the path of labeled frames under lp_dir
    #-----------------------------------------------------------------             
    for im in df_data.index:
        assert os.path.exists(os.path.join(lp_dir, im))
#         assert os.path.exists(os.path.join(dlc_dir, im))

    print(f"The number of labeled frames: {len(df_data.index)}")
    
    return df_data


# Modify https://github.com/danbider/lightning-pose/blob/7791c15ba77cc9d4ee0f754f513a3d1c11f42650/lightning_pose/utils/io.py#L23
def ckpt_path_from_base_path(
    base_path: str,
    model_name: str,
    version: int,
    logging_dir_name: str = "tb_logs/",
) -> str:
    """Given a path to a hydra output with trained model, extract the model .ckpt file.

    Args:
        base_path (str): path to a folder with logs and checkpoint. for example,
            function will search base_path/logging_dir_name/model_name...
        model_name (str): the name you gave your model before training it; appears as
            model_name in lightning-pose/scripts/config/model_params.yaml
        version (int. optional):
        logging_dir_name (str, optional): name of the folder in logs, controlled in
            train_hydra.py Defaults to "tb_logs/".
            
    Returns:
        str: path to model checkpoint

    """
    import glob
    
    model_dir = os.path.join(
        base_path,
        logging_dir_name,         
        model_name, # get the name string of the model (determined pre-training)
        )
    
    if version == None:
        # finding the most recent hydra path containing logging_dir_name
        model_search_path = os.path.join(
            model_dir,
            "version_*"
        )
        model_search_path = sorted(glob.glob(model_search_path))
        model_search_path = model_search_path[-1]  
    else:
        # finding the hydra path containing specific version
        model_search_path = os.path.join(
            model_dir,              
            "version_%i" % version,
        )
            
    model_search_path = os.path.join(
        model_search_path,
        "checkpoints",
        "*.ckpt",
    )
    # TODO: we're taking the last ckpt. make sure that with multiple checkpoints, this
    # is what we want
    model_ckpt_path = glob.glob(model_search_path)[-1]
    return model_ckpt_path



def predict_imgs_and_videos_in_dir(
    cfg: str, 
    ckpt_file: str, 
    LP_output_dir: str, 
    subclip_video_flag: bool=False) -> None:
    """ Perform prediction on images and videos, given config file, model specs and testing data.
    
    Args:
        cfg: config file 
        ckpt_file: a trained model
        LP_output_dir: path to save outputs
        subclip_video_flag: if true, make pridction on the subclip video; 
                            if false, make prediction on the original video

    Returns:
        generate the predictions and evaluation .csv files
    """
    # check if ckpt_file exist
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError("Cannot find checkpoint. Have you trained for too few epochs?")
    else:
        print(f"\nLoading the pretrained model: {ckpt_file}")

    assert os.path.isdir(LP_output_dir)
    
    # create data module
    cfg_pred = cfg.copy()
    cfg_pred.training.imgaug = "default"
    imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
    dataset_pred = get_dataset(
        cfg=cfg_pred, data_dir=cfg.data.data_dir, imgaug_transform=imgaug_transform_pred
    )
    data_module_pred = get_data_module(
        cfg=cfg_pred, dataset=dataset_pred, video_dir=cfg.data.video_dir
    )
    data_module_pred.setup()
    
    # ----------------------------------------------------------------------------------
    # predict on all labeled frames
    # ----------------------------------------------------------------------------------
    pretty_print_str("Predicting train/val/test images...")
    # compute and save frame-wise predictions
    preds_file = os.path.join(LP_output_dir, "predictions.csv")
    predict_dataset(
        cfg=cfg,
        data_module=data_module_pred,
        ckpt_file=ckpt_file,
        preds_file=preds_file,
    )
    # compute and save various metrics: pixel error and pca reprojection errors are included
    try:
        # TODO, LP is currently working on the labeled dataset that contains multiple dimensions. 
        compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module_pred)
    except Exception as e:
        print(f"Error computing metrics\n{e}")

    # ----------------------------------------------------------------------------------
    # predict folder of videos
    # ----------------------------------------------------------------------------------
    if cfg.eval.predict_vids_after_training:
        pretty_print_str("Predicting videos...")
        if cfg.eval.test_videos_directory is None:
            filenames = []
        else:
            filenames = check_video_paths(
                return_absolute_path(cfg.eval.test_videos_directory)
            )
            vidstr = "video" if (len(filenames) == 1) else "videos"
            pretty_print_str(
                f"Found {len(filenames)} {vidstr} to predict on {cfg.eval.test_videos_directory}"
            )

        testing_video_names = []
        
        # In this hackathon, we only make prediction on the first video to save running time.
        # To run prediction on all the videos under cfg.eval.test_videos_directory, change "filenames[:1]" to "filenames".
        for video_file in filenames[:1]:
            
            assert os.path.isfile(video_file)
            pretty_print_str(f"Predicting video: {video_file}...")

            # get the testing video names
            testing_video_names.append(video_file.split('/')[-1][:-4])
   
            # load the video
            video_clip = VideoFileClip(video_file) 
        
            #----------------------------------------------
            # Optional: make prediction on the subclip if the video is large to save time
            #----------------------------------------------
            if subclip_video_flag == True:
                # getting duration of the video 
                duration = video_clip.duration 
                print(f"Duration of the orginal video: {duration}") 

                # getting the subclip of the video, only use the first 30 seconds for testing
                # TODO: pass start_time, end_time as parameter instead of the first 30 seconds
                start_time = 0
                end_time   = 30
                video_subclip = video_clip.subclip(start_time, end_time)  
                duration = video_subclip.duration 
                print(f"Duration of the subclip video: {duration}") 

                # saving the subclip
                video_file = video_file.replace(".mp4", f".subclip{start_time}_{end_time}.mp4")
                print(f"Save subclip to: {video_file}")
                video_subclip.write_videofile(video_file)

            # ----------------------------------------------
            # update the image original dimension
            # ----------------------------------------------
            cfg_copy = copy.deepcopy(cfg)
            
            # ----------------------------------------------
            # get save name for prediction csv file
            # ----------------------------------------------
            video_pred_dir  = os.path.join(LP_output_dir, "video_preds")
            video_pred_name = os.path.splitext(os.path.basename(video_file))[0]
            prediction_csv_file = os.path.join(video_pred_dir, video_pred_name + ".csv")
            
            # ----------------------------------------------
            # get save name for labeled video csv file
            # ----------------------------------------------
            if cfg.eval.save_vids_after_training:
                labeled_vid_dir = os.path.join(video_pred_dir, "labeled_videos")
                labeled_mp4_file = os.path.join(
                    labeled_vid_dir, video_pred_name + "_labeled.mp4"
                )
            else:
                labeled_mp4_file = None
                
            # ----------------------------------------------  
            # predict on video: export predictions csv and a labeled video for a single video file.
            # ----------------------------------------------
            export_predictions_and_labeled_video(
                video_file=video_file,
                cfg=cfg_copy,
                ckpt_file=ckpt_file,
                prediction_csv_file=prediction_csv_file,
                labeled_mp4_file=labeled_mp4_file,
                data_module=data_module_pred,
                save_heatmaps=cfg.eval.get(
                    "predict_vids_after_training_save_heatmaps", False
                ),
            )
            # compute and save various metrics
            try:
                compute_metrics(
                    cfg=cfg_copy,
                    preds_file=prediction_csv_file,
                    data_module=data_module_pred,
                )
            except Exception as e:
                print(f"Error predicting on video {video_file}:\n{e}")
                continue

