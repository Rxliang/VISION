import sys
import os
import torch
from torch import optim
torch.manual_seed(3407)
from Sophia import SophiaG 

device = 'cuda'

class TrainManager:
    def __init__(self, train_left_dataloader, train_right_dataloader, test_left_dataloader, 
                 test_right_dataloader, LAVIS_model, predictor_model, metric_class):
        
        self.train_left_dataloader = train_left_dataloader
        self.train_right_dataloader = train_right_dataloader
        self.test_left_dataloader = test_left_dataloader
        self.test_right_dataloader = test_right_dataloader

        self.metric_class = metric_class
        self.LAVIS_model = LAVIS_model
        self.predictor_model = predictor_model
    
    def elementwise_corrcoef(self, x, y):
        assert x.shape == y.shape, "Input tensors must have the same shape"
        
        # Get the mean of x and y along the rows (dimension 0)
        x_mean = x.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        
        # Subtract the mean from x and y
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        # Compute the covariance
        covariance = (x_centered * y_centered).sum(dim=0, keepdim=True)
        
        # Compute the standard deviations of x and y
        x_std = x_centered.pow(2).sum(dim=0, keepdim=True).sqrt()
        y_std = y_centered.pow(2).sum(dim=0, keepdim=True).sqrt()
        
        # Compute the correlation coefficients
        corr_coeff = covariance / (x_std * y_std)
        
        return corr_coeff.squeeze()
            
    
    def train_MLP_one_epoch(self, dataloader, encoder, predictor, optimizer, criterion):

        total_loss = 0
        count = 0

        for img, sen, mri in dataloader:
            count += 1
            optimizer.zero_grad()
            sample = {"image": img, "text_input": list(sen)}

            encoder_output = encoder.extract_features(sample)
            encoder_output = encoder_output.multimodal_embeds
            # encoder_output = encoder_output.view(-1, 32, 32, 24)

            predict_output = predictor(encoder_output)
            loss = criterion(predict_output, mri.to(device))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / count
    
    def train_MLP_image_one_epoch(self, dataloader, encoder, predictor, optimizer, criterion):

        total_loss = 0
        count = 0

        for img, sen, mri in dataloader:
            count += 1
            optimizer.zero_grad()
            sample = {"image": img, "text_input": list(sen)}

            encoder_output = encoder.extract_features(sample, 'image')
            encoder_output = encoder_output.image_embeds
            # encoder_output = encoder_output.view(-1, 32, 32, 24)

            predict_output = predictor(encoder_output)
            loss = criterion(predict_output, mri.to(device))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / count

    def eval_MLP_model(self, dataloader, encoder, predictor):
        encoder.eval()
        predictor.eval()
        total_predict_result = []
        total_mri_result = []

        with torch.no_grad():
            for img, sen, mri in dataloader:
                sample = {"image": img, "text_input": list(sen)}

                encoder_output = encoder.extract_features(sample)
                encoder_output = encoder_output.multimodal_embeds
                # encoder_output = encoder_output.view(-1, 32, 32, 24)
                
                predict_output = predictor(encoder_output)

                total_predict_result.append(predict_output)
                total_mri_result.append(mri.to('cuda'))

        mri_correlation = self.elementwise_corrcoef(torch.vstack(total_predict_result), torch.vstack(total_mri_result))

        return mri_correlation
    
    def eval_MLP_image_model(self, dataloader, encoder, predictor):
        encoder.eval()
        predictor.eval()
        total_predict_result = []
        total_mri_result = []

        with torch.no_grad():
            for img, sen, mri in dataloader:
                sample = {"image": img, "text_input": list(sen)}

                encoder_output = encoder.extract_features(sample, 'image')
                encoder_output = encoder_output.image_embeds
                # encoder_output = encoder_output.view(-1, 32, 32, 24)
                
                predict_output = predictor(encoder_output)

                total_predict_result.append(predict_output)
                total_mri_result.append(mri.to('cuda'))

        mri_correlation = self.elementwise_corrcoef(torch.vstack(total_predict_result), torch.vstack(total_mri_result))

        return mri_correlation
    
    def train_transformer_one_epoch(self, dataloader, encoder, predictor, optimizer, criterion):
        total_loss = 0
        count = 0

        for img, sen, mri in dataloader:
            count += 1
            optimizer.zero_grad()
            sample = {"image": img, "text_input": list(sen)}
            encoder_output = encoder.extract_features(sample)
            predict_output = predictor(encoder_output.multimodal_embeds)
            loss = criterion(predict_output, mri.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / count
    
    def eval_transformer_model(self, dataloader, encoder, predictor):
        encoder.eval()
        predictor.eval()

        total_predict_result = []
        total_mri_result = []

        with torch.no_grad():
            for img, sen, mri in dataloader:

                sample = {"image": img, "text_input": list(sen)}

                encoder_output = encoder.extract_features(sample)
                predict_output = predictor(encoder_output.multimodal_embeds)
        

                total_predict_result.append(predict_output)
                total_mri_result.append(mri.to('cuda'))
                
          
        mri_correlation = self.elementwise_corrcoef(torch.vstack(total_predict_result), torch.vstack(total_mri_result))
                
        corr = self.metric_class.calculate_metric(mri_correlation)
        return corr


    def train(self, epoch, brain_type, lr, subj, save_path, train_type='image', model_type='MLP'):

        if brain_type == 'left':
            train_loader = self.train_left_dataloader
            test_loader = self.test_left_dataloader

        if brain_type == 'right':
            train_loader = self.train_right_dataloader
            test_loader = self.test_right_dataloader

        multimodal_encoder = self.LAVIS_model
        predictor = self.predictor_model

        multimodal_encoder.to(device)
        predictor.to(device)

        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.SmoothL1Loss(delta=1.0)
        # optimizer = optim.Adam(predictor.parameters(), lr, weight_decay=5e-4,)
        optimizer = SophiaG(predictor.parameters(), lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
        if model_type == 'MLP' and train_type == 'text':
            print('start multimodal image training')

            top3 = [(None,0)]*3
            top3_correlation_matrix = [(None,0)]*3

            for index in range(epoch):
                loss = self.train_MLP_one_epoch(train_loader, multimodal_encoder, predictor, optimizer, criterion)
                print('the total loss at epoch {} is {}'.format(index + 1, loss))

                correlation = self.eval_MLP_model(test_loader, multimodal_encoder, predictor)
                mean_correlation = self.metric_class.calculate_metric(correlation)

                if mean_correlation > top3[0][1]:
                    top3[0] = (predictor.state_dict(), mean_correlation)
                    top3.sort(key=lambda x: x[1],reverse=False)

                    top3_correlation_matrix[0] = (correlation.to('cpu'), mean_correlation)
                    top3_correlation_matrix.sort(key=lambda x: x[1],reverse=False)

                print('the median correlation test is {}'.format(mean_correlation))

            for i in range(3):
                torch.save(top3[i][0], os.path.join(save_path, f"top{i+1}_model_subj{subj}_{brain_type}_corr{top3[i][1]}.pt"))
                torch.save(top3_correlation_matrix[i][0], os.path.join(save_path, f"top{i+1}_corr_matrix_subj{subj}_{brain_type}_{top3[i][1]}.pt"))

        if model_type == 'MLP' and train_type == 'image':
            top3 = [(None,0)]*3
            top3_correlation_matrix = [(None,0)]*3

            print('start pure image training')

            for index in range(epoch):
                loss = self.train_MLP_image_one_epoch(train_loader, multimodal_encoder, predictor, optimizer, criterion)
                print('the total loss at epoch {} is {}'.format(index + 1, loss))

                correlation = self.eval_MLP_image_model(test_loader, multimodal_encoder, predictor)
                mean_correlation = self.metric_class.calculate_metric(correlation)

                if mean_correlation > top3[0][1]:
                    top3[0] = (predictor.state_dict(), mean_correlation)
                    top3.sort(key=lambda x: x[1],reverse=False)

                    top3_correlation_matrix[0] = (correlation.to('cpu'), mean_correlation)
                    top3_correlation_matrix.sort(key=lambda x: x[1],reverse=False)

                print('the median correlation test is {}'.format(mean_correlation))

            for i in range(3):
                torch.save(top3[i][0], os.path.join(save_path, f"top{i+1}_model_subj{subj}_{brain_type}_corr{top3[i][1]}.pt"))
                torch.save(top3_correlation_matrix[i][0], os.path.join(save_path, f"top{i+1}_corr_matrix_subj{subj}_{brain_type}_{top3[i][1]}.pt"))


