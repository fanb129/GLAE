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

// ==========================================
// LSTM + DANN + GLAE (Refactored Version)
// Folder: lstm_dann_glae/
// ==========================================

// ---------------- SEED Dataset ----------------

// 1. Run Step 1 Only (Pretraining)
// Output: lstm_dann_glae/seed3_step1.log
setsid nohup python -u lstm_dann_glae/main.py --dataset_name="seed3" --step1 --cuda="0" > lstm_dann_glae/seed3_step1.log 2>&1 &

// 2. Run Step 2 Only (GLA with Anchor Selection)
// Note: Requires pre-trained models from Step 1
// Output: lstm_dann_glae/seed3_step2.log
setsid nohup python -u lstm_dann_glae/main.py --dataset_name="seed3" --step2 --cuda="1" > lstm_dann_glae/seed3_step2.log 2>&1 &

// 3. Run Full Pipeline (Step 1 + Step 2 Sequentially)
// Recommended for clean run
// Output: lstm_dann_glae/seed3_full.log
setsid nohup python -u lstm_dann_glae/main.py --dataset_name="seed3" --step1 --step2 --cuda="1" > lstm_dann_glae/seed3_full.log 2>&1 &


// ---------------- SEED-IV Dataset ----------------

// 1. Run Step 1 Only (Pretraining)
setsid nohup python -u lstm_dann_glae/main.py --dataset_name="seed4" --step1 --cuda="1" > lstm_dann_glae/seed4_step1.log 2>&1 &

// 2. Run Step 2 Only (GLA with Anchor Selection)
setsid nohup python -u lstm_dann_glae/main.py --dataset_name="seed4" --step2 --cuda="1" > lstm_dann_glae/seed4_step2.log 2>&1 &

// 3. Run Full Pipeline (Step 1 + Step 2 Sequentially)
setsid nohup python -u lstm_dann_glae/main.py --dataset_name="seed4" --step1 --step2 --cuda="1" > lstm_dann_glae/seed4_full.log 2>&1 &