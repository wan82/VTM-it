This VTM using csv as prediction guider to let VTM be predict:
Also it is a very Immature software. Didn’t follow with standard  develop rule, just can run, Please forgive me >_<
This VTM project is using outside csv file to accelerate the encoding
===============================================

The Core change in source/Lib/EncoderLib/EncCU.cpp
You can use WAN_CHANGE to target which part is my code

Other reference code change in 
/VVCSoftware_VTM-128-20.0/source/Lib/EncoderLib/EncCfg.h 
/VVCSoftware_VTM-128-20.0/source/App/EncoderApp/EncAppCfg.h
/VVCSoftware_VTM-128-20.0/source/App/EncoderApp/EncAppCfg.cpp
/VVCSoftware_VTM-128-20.0/source/App/EncoderApp/EncApp.cpp
/VVCSoftware_VTM-128-20.0/source/Lib/EncoderLib/EncSlice.cpp *

These changes not very important, just for read csv file and input as parameters. 
* EncSlice.cpp change code is: you can End encoding in advance at the frame you specify

You can use: Ctrl + Shift  + F (MacOS: cmd + shift + F) find all WAN_CHANGE in viscose

================================================

For the csv file, you can find the csv pattern in file: predictionCsv.zip
dataFrame like:
-----------------------------------------------------------------------------------
name		|QP	|QP_Value	|FrameID	|CTU_ID		|predicted
-----------------------------------------------------------------------------------
BQMall.csv	|22	|27		|4		|0		|0
-----------------------------------------------------------------------------------
Each row stand for 1 CU(128x128) in a frame
CTU_ID follows from top to bottom and from left to right to locate each 128-level CU in the frame
So, it can position any 128-level CU using [FrameID, CTU_ID]

Actually: Column [name, QP, QP_Value] these 3 is unnecessary, I use it on debug, but forget to remove, 
You can remove it as you prefer


The command to start up Encoder is like this: 
/home/wan/Dev/VTM/VVCSoftware_VTM-VTM-20.0/bin/EncoderAppStatic -c /home/wan/Dev/VTM/VVCSoftware_VTM-VTM-20.0/cfg/encoder_randomaccess_vtm.cfg -c /mnt/data/wan_data/videoProcess/per-sequence/PartyScene.cfg -i /mnt/data/wan_data/wan_yuv/PartyScene_832x480_50.yuv --QP=22 --BitstreamFile="/dev/null" --ReconFile="/dev/null" --SplitFilePath="/mnt/data/wan_data/CsvData/NB_predictionFull/PartyScene_22.csv" > /mnt/data/wan_data/wan_logs/fullFrame_NB/PartyScene_22.log

The new Parameter for bash command is: —SplitFilePath=“Your_Prediction_guide_file.csv”
You need to find target prediction csv file for target encoding video

All the prediction csv file in predictionCsv.zip
These files use model Naive Bayes to predict the QT split flag
======================================================


If you still have any questions need to ask, please contact me, my email is: xiangyu.wan@alumnos.upm.es
^_^
