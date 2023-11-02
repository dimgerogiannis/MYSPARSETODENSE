dependencies:
--------------
torch 1.9.0+cu102
numpy
scipy
scikit-learn
matplotlib
tqdm
trimesh

-------------------------------------------------------------------------------
To Reproduce our results on CoMA for both identity and expressions split protocol, please follow these steps:
1) Please download the CoMA dataset from https://flame.is.tue.mpg.de/ and include it in the CoMA_dataset folder.
Please find a Readme file inside ./template folder that specify more files that you need to download.
2) Gererate processed data using the command below, while specifying the split (Id or Expr) protocol and the fold (from 1 to 4):
       python Data_processing.py --data_path="./CoMA_dataset/" --Split="Id"  --test_fold=1

3) Run the S2D-Dec model after specifying the split protocol and the desired fold (from 1 to 4):
       python S2D-Dec.py --Split="Id"  --test_fold=1

4) After running the 4 folds, you can compute the cross validation error:
       python Err_CrossVal.py --Split="Id" 

------------------------------------------------------------------------------------

To visualize some Motion3DGAN samples:
1) Generate the preprocessed data by specifying the desired label (from 0 to 11) and identity (from 0 to 11).
      python data_generationMotion3DGAN.py --label=0 --id=0 

2) Run the S2D-Dec model to show the sequence (30 meshes) using the same label and id above:  
      python S2D-Dec.py --label=0 --Lands="Motion3DGAN"

The resulting meshes will be included in Results\Motion3DGAN\sample_"label"\predicted_meshes


---------------------------------------------------------------------------------
The training code of Motion3DGAN and S2D-Dec will be released upon publication


-------------------------------------------------------------------------------
Parts of this code are based on the code from: 
https://github.com/gbouritsas/Neural3DMM/
https://github.com/anuragranj/coma

Some data included in the template folder is taken from: https://github.com/TimoBolkart/TF_FLAME/tree/master/data
