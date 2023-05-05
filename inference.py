import sys
import os
import torch
import numpy as np
import yaml
import argparse
from Model.mlp import MLP_model
from Data.dataset import MRI_test_dataset
from lavis.models import load_model_and_preprocess
from torch.utils.data import DataLoader

device = 'cuda'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    return args

def inference_MLP_image_model(dataloader, encoder, predictor):
    encoder.eval()
    predictor.eval()
    total_predict_result = []

    with torch.no_grad():
        for img, sen in dataloader:
            sample = {"image": img, "text_input": list(sen)}

            encoder_output = encoder.extract_features(sample, 'image')
            encoder_output = encoder_output.image_embeds
            encoder_output = encoder_output.view(-1, 32, 32, 24)
            
            predict_output = predictor(encoder_output)

            total_predict_result.append(predict_output)
        
        total_predict_result = torch.vstack(total_predict_result)
    
    return total_predict_result.to('cpu')

def inference_MLP_multimodal_model(dataloader, encoder, predictor):
    encoder.eval()
    predictor.eval()
    total_predict_result = []

    with torch.no_grad():
        for img, sen in dataloader:
            sample = {"image": img, "text_input": list(sen)}

            encoder_output = encoder.extract_features(sample)
            encoder_output = encoder_output.multimodal_embeds
            encoder_output = encoder_output.view(-1, 32, 32, 24)
            
            predict_output = predictor(encoder_output)

            total_predict_result.append(predict_output)

        total_predict_result = torch.vstack(total_predict_result)
    
    return total_predict_result.to('cpu')

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
    model_save_path = config['Dataset']['model_save_path']
    train_type = config['Dataset']['train_type']
    subject_submission_dir = config['Dataset']['subject_submission_dir']
    
    channels = config['MLP_model']['channels']
    patch_size = config['MLP_model']['patch_size']
    dim = config['MLP_model']['dim']
    depth = config['MLP_model']['depth']

    blip2_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", 
    model_type="pretrain", is_eval=True, device=device)

    test_dataset = MRI_test_dataset(subj, data_type, brain_type, vis_processors, txt_processors, data_dir, csv_file_path)
    test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
    feature_size = test_dataset.mri_dim

    MLP_model_class = MLP_model(channels, patch_size, dim, depth, feature_size)
    predictor = MLP_model_class.init_MLP_Mixer()
    predictor.load_state_dict(torch.load(model_save_path))

    blip2_model.to(device)
    predictor.to(device)

    if train_type == 'image':
        if brain_type == 'left':
            bt = 'lh'
        if brain_type == 'right':
            bt = 'rh'
        total_predict_result = inference_MLP_image_model(test_loader, blip2_model, predictor)
        np.save(os.path.join(subject_submission_dir, bt + '_pred_test.npy'), total_predict_result.numpy())
    
    if train_type == 'text':
        if brain_type == 'left':
            bt = 'lh'
        if brain_type == 'right':
            bt = 'rh'
        total_predict_result = inference_MLP_multimodal_model(test_loader, blip2_model, predictor)
        np.save(os.path.join(subject_submission_dir, bt + '_pred_test.npy'), total_predict_result.numpy())

if __name__=='__main__':
    main()