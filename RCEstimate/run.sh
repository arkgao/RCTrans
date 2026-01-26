# train the network
python train.py --opt configs/train.yaml

# test the network
python test.py --opt configs/test_validation.yaml

# test for reconstruction
python test_recon.py --input_dir ../TransRecon/exp/bowl/pred_correspondence/input_data --output_dir ../TransRecon/exp/bowl/pred_correspondence/raw_output --opt configs/test_recon.yaml