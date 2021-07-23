config=$1
model=$2
root_folder=$3
python scripts/main.py \
        --config-file $config \
        --root  $root_folder \
        model.load_weights $model \
        test.evaluate True
