import sys
import torch
torch.manual_seed(3407)

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
    
    def train_one_epoch(self, dataloader, encoder, predictor, optimizer, criterion):
        
        total_loss = 0
        count = 0

        for img, sen, mri in dataloader:
            count += 1
            optimizer.zero_grad()
            sample = {"image": img, "text_input": list(sen)}
            encoder_output = encoder.extract_features(sample)
            predict_output = predictor(encoder_output.multimodal_embeds)

            #encoder_output = encoder.extract_features(sample, mode="image")
            #predict_output = predictor(encoder_output.image_embeds)
            
            loss = criterion(predict_output, mri.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / count
    

    
    def eval_model_mse(self, dataloader, encoder, predictor):
        
        encoder.eval()
        predictor.eval()

        total_predict_result = []
        total_mri_result = []

        with torch.no_grad():
            for img, sen, mri in dataloader:

                sample = {"image": img, "text_input": list(sen)}
                encoder_output = encoder.extract_features(sample)
                predict_output = predictor(encoder_output.multimodal_embeds)

                #encoder_output = encoder.extract_features(sample, mode="image")
                #predict_output = predictor(encoder_output.image_embeds)
        

                total_predict_result.append(predict_output)

                total_mri_result.append(mri.to('cuda'))

        mri_correlation = self.elementwise_corrcoef(torch.vstack(total_predict_result), torch.vstack(total_mri_result))
                
        return torch.mean(mri_correlation)

    def train(self, epoch, brain_type, lr):
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
        optimizer = optim.Adam(predictor.parameters(), lr)

        total_loss = 0
        for index in range(epoch):
            loss = self.train_one_epoch(train_loader, multimodal_encoder, predictor, optimizer, criterion)

            sys.stderr.write('the total loss at epoch {} is {}'.format(index + 1, loss))

            correlation = self.eval_model_mse(test_loader, multimodal_encoder, predictor)
            sys.stderr.write('the mean correlation test is {}'.format(correlation))