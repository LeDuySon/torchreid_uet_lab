configs=$1
models=$2
save_dir=$3
epoch=15
eval_freq=5

python scripts/main.py --config-file $configs -s "uet_reid" -t "uet_reid" --transforms random_flip --root "reid-data" loss.name "triplet" sampler.train_sampler RandomIdentitySampler model.load_weights $models  data.save_dir $save_dir train.max_epoch $epoch test.eval_freq $eval_freq  

#https://github.com/KaiyangZhou/deep-person-reid/issues/155#issuecomment-484363594_


