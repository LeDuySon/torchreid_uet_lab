configs=$1
models=$2
python scripts/main.py --config-file $configs -s "uet_reid" -t "uet_reid" --transforms random_flip --root "reid-data" model.load_weights $models loss.name "triplet"



