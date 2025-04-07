### 1. How to train

  1.1 clone this repository to local
  
  1.2 download dataset from google drive https://drive.google.com/drive/folders/1n7jWwbzueo9ESeArs1OcIla9X_VbNh7v?dmr=1&ec=wgc-drive-hero-goto
  
  1.3 put dataset in to this repo, with same level with train.py, test.py, etc.
  
  1.4 install environment with conda env create -f environment.yml
  
  1.5 train model with command python train.py --learning_rate 1e-4 --activate_func sigmoid --hidden_dim 2304 --l2_lambda 1e-6 --batch_size 10, model now only support activate_func [sigmoid, relu], you can also train with other hyperparameters.
  
  1.6 after training, loss curve will be writen to Training_validation_loss.jpg, validation accuracy will be writen to Training_validation_accuracy.jpg.

### 2. How to test model 

  2.1 test model with command python test.py --model path_to_model.pkl

### 3. How to visualize model weights

   3.1 visualize with command python visual_model.py --model path_to_model.pkl
