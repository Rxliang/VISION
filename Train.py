import sys
import torch
from torch import optim
torch.manual_seed(3407)

device = 'cuda'

class TrainManager:
    def __init__(self, train_left_dataloader, train_right_dataloader, test_left_dataloader, 
                 test_right_dataloader, LAVIS_model, predictor_model):
        
        self.train_left_dataloader = train_left_dataloader
        self.train_right_dataloader = train_right_dataloader
        self.test_left_dataloader = test_left_dataloader
        self.test_right_dataloader = test_right_dataloader

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
            encoder_output = encoder_output.view(-1, 32, 32, 24)

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
                encoder_output = encoder_output.view(-1, 32, 32, 24)
                
                predict_output = predictor(encoder_output)
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
                
        return mri_correlation


    def train(self, epoch, brain_type, lr, model_type='MLP'):

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

        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(predictor.parameters(), lr, weight_decay=5e-4,)

        if model_type == 'MLP':
            top3 = [(None,0)]*3
            for index in range(epoch):
                loss = self.train_MLP_one_epoch(train_loader, multimodal_encoder, predictor, optimizer, criterion)
                sys.stderr.write('the total loss at epoch {} is {}'.format(index + 1, loss))

                correlation = self.eval_MLP_model(test_loader, multimodal_encoder, predictor)
                mean_correlation = torch.mean(correlation)

                if mean_correlation > top3[0][1]:
                    top3[0] = (predictor.state_dict(), mean_correlation)
                    top3.sort(key=lambda x: x[1],reverse=True)
                sys.stderr.write('the mean correlation test is {}'.format(mean_correlation))

            for i in range(3):
                torch.save(top3[i][0],f"top{i+1}_model_corr{top3[i][1]}.pt")

        if model_type == 'transformer':
            for index in range(epoch):
                loss = self.train_MLP_one_epoch(train_loader, multimodal_encoder, predictor, optimizer, criterion)
                sys.stderr.write('the total loss at epoch {} is {}'.format(index + 1, loss))

                correlation = self.eval_MLP_model(test_loader, multimodal_encoder, predictor)
                mean_correlation = torch.mean(correlation)
                sys.stderr.write('the mean correlation test is {}'.format(mean_correlation))

