configs = $1

python scripts/main.py --config-file $configs --transforms random_flip --root "reid-data"


