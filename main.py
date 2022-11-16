from data_loader import WifiDataset, FMCWDataset, UWBDataset, OurDataset
import tqdm
import config as cfg
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch
import torch.nn as nn
import dual as base_model     
from builder import Builder    




if cfg.data == 'WIFI':        
    data = WifiDataset(batch_size=cfg.batch_size, 
                            scene_per_batch=cfg.scene_per_batch, 
                            k_shots = cfg.k_shots, 
                            seed=cfg.seed)
elif cfg.data == 'UWB':
    data = UWBDataset(batch_size=cfg.batch_size, 
                            scene_per_batch=cfg.scene_per_batch, 
                            k_shots = cfg.k_shots, 
                            seed=cfg.seed)    
elif cfg.data == 'FMCW':
    data = FMCWDataset(batch_size=cfg.batch_size, 
                            scene_per_batch=cfg.scene_per_batch, 
                            k_shots = cfg.k_shots, 
                            seed=cfg.seed)     
elif cfg.data == "OURS":
    data = OurDataset(batch_size=cfg.batch_size, 
                            scene_per_batch=cfg.scene_per_batch, 
                            k_shots = cfg.k_shots, 
                            seed=cfg.seed)
    
cuda = True if torch.cuda.is_available() else False 

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

builder = Builder(base_model, cuda, Tensor, data)

best_val_acc = 0.0
builder._enabel_cuda()

# with tqdm.tqdm(total=cfg.total_train_batches) as pbar_e:
    
#     for e in range(cfg.total_epochs):        

#         total_c_loss, total_accuracy = builder.run_tuning_epoch(cfg.total_train_batches, 'train', mode = 'train')
#         print("Epoch {}: ft_loss:{} ft_accuracy:{}".format(e, total_c_loss, total_accuracy))


#         total_c_loss, total_accuracy = builder.run_training_epoch(cfg.total_train_batches, data_type= "train")
#         print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
        

# builder.save_model(path= "model_1_shot.pth")


builder = Builder(base_model, cuda, Tensor, data)
builder.load_model(path= "model_{}_shot.pth".format(cfg.k_shots))
builder._enabel_cuda()
print("======= env1 fine-tuning ===== ")
with tqdm.tqdm(total=cfg.total_train_batches) as pbar_e:
    
    for e in range(cfg.total_epochs):

        total_c_loss, total_accuracy = builder.run_tuning_epoch(cfg.total_val_batches, 'val', mode= "train")
        
        # total_val_c_loss, total_val_accuracy = builder.run_training_epoch(cfg.total_val_batches, data_type= "val")
        # print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))

builder.save_model(path= "model_middle_{}_shot.pth".format(cfg.k_shots))


for i in range (2, 5):
    data = OurDataset(batch_size=cfg.batch_size, 
                                scene_per_batch=cfg.scene_per_batch, 
                                k_shots = cfg.k_shots, 
                                seed=cfg.seed,
                                envs = ["env{}".format(i)]
                                )
    cfg.num_class = cfg.num_new_class
    builder = Builder(base_model, cuda, Tensor, data)
    builder.load_model_for_testing( path= "model_middle_{}_shot.pth".format(cfg.k_shots))
    builder._enabel_cuda()
    print("======= env{} ===== ".format(i))
    with tqdm.tqdm(total=cfg.total_train_batches) as pbar_e:
        
        for e in range(cfg.total_epochs):

            # total_c_loss, total_accuracy = builder.run_tuning_epoch(cfg.total_test_batches, 'test', mode = "test")

            total_test_c_loss, total_test_accuracy, std = builder.run_test_epoch(cfg.total_test_batches)
            print("Epoch {}: test_loss:{} test_accuracy:{} std:{}".format(e, total_test_c_loss, total_test_accuracy, std))
