import torch



class Config:

    target_columns = ['bowel_injury','extravasation_injury',
                      'kidney_healthy', 'kidney_low', 'kidney_high',
                      'liver_healthy', 'liver_low', 'liver_high',
                      'spleen_healthy','spleen_low', 'spleen_high','any_injury']
    ckp = torch.load('logs/additional_info.pth')
    valid_ckp = torch.load('logs/best_valid_score.pth')
    # last_epoch = ckp['epoch']
    # model_weights = ckp['model']
    # optim_dict = ckp['optimizer']
    # best_loss = ckp['best_loss']
    # best_log_score = ckp['best_log_score']


Config = Config()

#print(Config.ckp)