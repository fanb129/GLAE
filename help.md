setsid nohup python -u lstm_baseline/seed3_train.py > lstm_baseline/seed3_train.log 2>&1 &
setsid nohup python -u lstm_baseline/seed4_train.py > lstm_baseline/seed4_train.log 2>&1 &


setsid nohup python -u lstm_glae/seed3_train.py > lstm_glae/seed3_train.log 2>&1 &
setsid nohup python -u lstm_glae/seed4_train.py > lstm_glae/seed4_train.log 2>&1 &

// dmmr_glae
setsid nohup python -u dmmr_glae/main.py --dataset_name="seed3" > dmmr_glae/seed3_train.log 2>&1 &
setsid nohup python -u dmmr_glae/main.py --dataset_name="seed4" > dmmr_glae/seed4_train.log 2>&1 &

// dmmr_baseline
setsid nohup python -u dmmr_baseline/main.py --dataset_name="seed3" > dmmr_baseline/seed3_train.log 2>&1 &
setsid nohup python -u dmmr_baseline/main.py --dataset_name="seed4" > dmmr_baseline/seed4_train.log 2>&1 &

// lstm_dann
setsid nohup python -u lstm_dann/main.py --dataset_name="seed3" > lstm_dann/seed3_train.log 2>&1 &
setsid nohup python -u lstm_dann/main.py --dataset_name="seed4" > lstm_dann/seed4_train.log 2>&1 &

//lstm_dann_glae
setsid nohup python -u lstm_dann/main_glae.py --dataset_name="seed3" --cuda="2" > lstm_dann/seed3_glae.log 2>&1 &
setsid nohup python -u lstm_dann/main_glae.py --dataset_name="seed4" --cuda="3" > lstm_dann/seed4_glae.log 2>&1 &