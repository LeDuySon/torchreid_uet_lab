configs=$1
models=$2
save_dir=$3
python scripts/main.py --config-file $configs -s "uet_reid" -t "uet_reid" --transforms random_flip --root "reid-data" model.load_weights $models loss.name "softmax" data.save_dir $save_dir



