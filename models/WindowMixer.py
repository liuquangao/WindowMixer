import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.RevIN import RevIN
import math

def get_activation(activation):
    if activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        print('None activation')
        return nn.Identity()
    
class MlpMixerBlock(nn.Module):
    def __init__(self, d_model=128, seq_len=336, dropout=0.1, activation='tanh'):
        super(MlpMixerBlock, self).__init__()
        
        self.norm_all_1 = nn.LayerNorm([seq_len, d_model])
        self.norm_all_2 = nn.LayerNorm([seq_len, d_model])

        # 特征维度MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            get_activation(activation),
            nn.Linear(d_model, d_model, bias=True),
            nn.Dropout(dropout),
        )
        
        # 序列维度MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len, bias=True),
            get_activation(activation),
            nn.Linear(seq_len, seq_len, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, src):                                         # src: [bsz, nvar, seq_len, d_model]
        
        x = self.feature_mlp(src) + src                             # x: [bsz, nvar, seq_len, d_model]
        x = self.norm_all_1(x)                                      # x: [bsz, nvar, seq_len, d_model]

        x = self.time_mlp(x.permute(0,1,3,2)).permute(0,1,3,2) + x  # x: [bsz, nvar, seq_len, d_model]
        x = self.norm_all_2(x)                                      # x: [bsz, nvar, seq_len, d_model]
      
        return x

# window embedding
class WindowEmbed(nn.Module):
    def __init__(self, nvar=1, d_model=8, w_sizes=8, seq_len=336, individual=False):
        super().__init__()
        
        self.window = w_sizes
        self.context_len = 2 * self.window + 1
        self.individual = individual
        self.d_model = d_model

        if individual:
            # 每个变量一个线性层，手动初始化权重和偏置
            self.W_Ps = nn.Parameter(torch.Tensor(nvar, self.context_len, d_model))
            self.biases = nn.Parameter(torch.Tensor(nvar, d_model))
            k = 1.0 / math.sqrt(self.context_len)
            nn.init.uniform_(self.W_Ps, -k, k)
            nn.init.uniform_(self.biases, -k, k)
        else:
            # 所有变量共享一个线性层
            self.W_P = nn.Linear(self.context_len, d_model, bias=True)

    def forward(self, x):                                           # x: [bsz, nvar, seq_len]
        bsz, nvar, seq_len = x.shape

        # 提取首尾边缘值并连接到原始张量的两端
        left_edge = x[:, :, :1].expand(-1, -1, self.window)
        right_edge = x[:, :, -1:].expand(-1, -1, self.window)
        x = torch.cat([left_edge, x, right_edge], dim=-1)           # x: [bsz, nvar, seq_len + 2 * window]

        # 按窗口大小展开
        x = x.unfold(dimension=-1, size=self.context_len, step=1)   # x: [bsz, nvar, seq_len, context_len]

        if self.individual:
            x = torch.einsum('bnij,njk->bnik', x, self.W_Ps)        # x: [bsz, nvar, seq_len, d_model]
            x = x + self.biases.view(1, nvar, 1, self.d_model)      # 加上偏置
        else:
            x = self.W_P(x)                                         # x: [bsz, nvar, seq_len, d_model]

        return x  # x: [bsz, nvar, seq_len, d_model]

# 不同的任务有不同的头 目前只考虑预测任务
class FlattenHead(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, task_name, head_dropout = 0.1,individual=False,num_class=None, nvar=None):
        super().__init__()
        self.task_name = task_name
        self.individual = individual
        if task_name == 'long_term_forecast' or task_name == 'short_term_forecast' or\
        task_name == 'imputation' or task_name == 'anomaly_detection':
            
            if self.individual:
                # 每个变量一个线性层，手动初始化权重和偏置
                self.weight = nn.Parameter(torch.Tensor(nvar, seq_len * d_model, pred_len))
                self.bias = nn.Parameter(torch.Tensor(nvar, pred_len))
                k = 1.0 / math.sqrt(seq_len * d_model)
                nn.init.uniform_(self.weight, -k, k)
                nn.init.uniform_(self.bias, -k, k)
                self.dropout = nn.Dropout(head_dropout)
            else:
                # 所有变量共享一个线性层
                self.layer1 = nn.Linear(seq_len * d_model, pred_len)
                self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                                           # x: [bsz, nvar, seq_len, d_model]
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or \
        self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            
            if self.individual:
                x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))  # x: [bsz, nvar, seq_len*d_model]
                x = torch.einsum('bni,nij->bnj', x, self.weight) + self.bias        # x: [bsz, nvar, pred_len]
                x = self.dropout(x)
                x = x.permute(0,2,1)                                                # x: [bsz, pred_len, nvar]
            
            else:
                x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))  # x: [bsz, nvar, seq_len*d_model]
                x = self.dropout(self.layer1(x))                                    # x: [bsz, nvar, pred_len]
                x = x.permute(0,2,1)                                                # x: [bsz, pred_len, nvar]                                                                            
        return x

class Model(nn.Module):
    def __init__(self, configs, revin=True):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        if configs.individual:
            self.individual = True
        else:
            self.individual = False
        self.channels = configs.enc_in

        # norm
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        # window embedding
        self.patch_embedding = WindowEmbed(nvar=configs.enc_in,d_model=configs.d_model,w_sizes=configs.w_size, individual=self.individual)
    
        self.mix_blocks = nn.ModuleList([MlpMixerBlock(d_model=configs.d_model, seq_len=self.seq_len, activation=configs.activation)
                                     for i in range(configs.e_layers)])
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(seq_len=self.seq_len, pred_len=configs.pred_len,
                                    d_model=configs.d_model, task_name=self.task_name, nvar=self.channels)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    #TODO:目前还不知道先分解好还是先归一化好
    #TODO:趋势项现在只是一个线性层,是否需要更复杂的处理
    def forecast(self, x):
        
        # norm
        if self.revin: 
            x = self.revin_layer(x, 'norm')

        seasonal, trend = self.decompsition(x)

        x = seasonal

        x = x.permute(0,2,1)                                            # x: [bsz, nvar, seq_len] 
        x = self.patch_embedding(x)                                     # x: [bsz, nvar, seq_len, d_model] 

        for mix_block in self.mix_blocks:
            x = mix_block(x)                                            # x: [bsz, nvar, seq_len, d_model]

        x = self.head(x)                                                # x: [bsz, pred_len, nvar]

        trend = self.linear_trend(trend.permute(0,2,1)).permute(0,2,1)
        x = x + trend

        # denorm
        if self.revin: 
            x = self.revin_layer(x, 'denorm')

        return x 

    ## TODO:待完善
    def imputation(self, x, x_mark_enc, x_dec, x_mark_dec, mask):
        
        return x 


    # TODO:待完善
    def anomaly_detection(self, x):
        return x

    
    # TODO:待完善
    def classification(self, x):
        return x 
 

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None