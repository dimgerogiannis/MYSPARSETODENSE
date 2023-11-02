
Please find a Readme file inside ./Data/template folder that specify some files that you need to download to that folder

1) Download the CoMA dataset from https://coma.is.tue.mpg.de/

2) Create landmarks data from COMA dataset with CoMA_CreateLnadmarks_data.py function
The file ./Data/COMA_data_list.txt provides the list of CoMA subsequences used in the training

3) transform landmarks to SRVF data with SRVF_Process/create_SRVF_COMA.m function

4) start training with SRVF_training/main.py (after fixing some parametres and paths to data)

./SRVF_Process folder contains dynamic_transfert_SRVF.m function that can apply SRVF motion to neutral landmarks configuration to get a 
sequence of landmarks.
