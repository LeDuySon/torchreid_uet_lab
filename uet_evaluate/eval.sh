name=$1 # save name 
data_path=$2 
model_path=$3

reid_model_name="osnet_ain_x1_0"


python get_features.py --data-dir $data_path --reid_model_path $model_path --reid_model_name $reid_model_name --save-dir predictions --save-name $name 


