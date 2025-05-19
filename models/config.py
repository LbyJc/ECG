import torch
# overall
batch_size = 128#64
epochs = 100
lr = 0.001
random_state = 2
k_folds = 5
outf = 'logs_LLOid2_oldModel_BadFileDeleted_run2_BS128'
num_workers = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
attn_mode = 'airECG_catFusion_newTCN'
data_dir = 'Dataset_Aligned_BadFileDeleted/'  # 图像文件夹
val_dir = 'Dataset_Aligned_BadFileDeleted_test/'
test_dir = 'Dataset_Aligned_BadFileDeleted_test/'
path = outf+'/best_model.pth'
