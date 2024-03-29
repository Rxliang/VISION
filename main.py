import os
import sys
import yaml
import torch
import argparse
from Model.mlp import MLP_model,Siren,MLP_basic
from Train import TrainManager
from lavis.models import load_model_and_preprocess
from Data.dataset import MRI_dataset, train_test_split, noisy_celing_metric

device = 'cuda'
torch.manual_seed(3407)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    return args

def main():

    args = parse_arguments()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    subj = config['Dataset']['subj']
    data_type = config['Dataset']['data_type']
    data_dir = config['Dataset']['data_dir']
    batch_size = config['Dataset']['batch_size']
    csv_file_path = config['Dataset']['csv_file_path']
    brain_type = config['Dataset']['brain_type']
    model_save_folder = config['Dataset']['model_save_folder']
    train_type = config['Dataset']['train_type']

    channels = config['MLP_model']['channels']
    patch_size = config['MLP_model']['patch_size']
    dim = config['MLP_model']['dim']
    depth = config['MLP_model']['depth']

    epoch = config['parameter']['epoch']
    lr = config['parameter']['lr']

    sys.stderr.write('start loading model')
    blip2_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", 
    model_type="base", is_eval=True, device=device)

    train_dataset = MRI_dataset(subj, data_type, brain_type, vis_processors, txt_processors, data_dir, csv_file_path)
    feature_size = train_dataset.mri_dim
    train_loader, eval_loader, test_loader = train_test_split(train_dataset, batch_size)
    predictor = MLP_basic(feature_size,160)
    # predictor = Siren(feature_size,100,1024,3,30)
    # MLP_model_class = MLP_model(channels, patch_size, dim, depth, feature_size)
    # predictor = MLP_model_class.init_Siren()

    nc_class = noisy_celing_metric(data_dir, subj, brain_type)

    trainer = TrainManager(train_loader, train_loader, test_loader, test_loader, blip2_model, predictor, nc_class)

    print('start training')
    subj_save = format(subj, '02')
    trainer.train(epoch, brain_type, lr, subj_save, model_save_folder, train_type)

if __name__=='__main__':
    main()