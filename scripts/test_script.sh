config=$1
models=$2
#root_folder=$3
python scripts/test.py --config-file $config -s "uet_reid" -t "uet_reid" --root "reid-data" test.evaluate True model.load_weights $models test.dist_metric cosine 
