import os
import yaml
import torch
import argparse
from Model.mlp import MLP_basic
from Trainer import TrainManager
from lavis.models import load_model_and_preprocess
from Data.dataset import MRI_dataset, train_test_split

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



    input_size = config['MLP_model']['input_size']
    output_size = config['MLP_model']['output_size']
    hidden_size = config['MLP_model']['hidden_size']
    num_layers = config['MLP_model']['num_layers']

    epoch = config['parameter']['epoch']
    lr = config['parameter']['lr']

    blip2_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", 
    model_type="pretrain", is_eval=True, device=device)
    
    predictor = MLP_basic(input_size, output_size, hidden_size, num_layers)

    train_dataset = MRI_dataset(subj, data_type, brain_type, vis_processors, txt_processors, data_dir, csv_file_path)

    train_loader, eval_loader, test_loader = train_test_split(train_dataset, batch_size)

    trainer = TrainManager(train_loader, train_loader, test_loader, test_loader, blip2_model, predictor)
    trainer.train(epoch, 'left', lr)

if __name__=='__main__':
    main()