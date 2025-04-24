import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class patch_embedding(nn.Module):
    def __init__(self,patch_dim,embedding_dim):
        super().__init__()
        self.patch_dim=patch_dim
        self.embedding=nn.Sequential(
            nn.Linear(patch_dim,embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def slice(self,x):
        x=x.view(x.size(0),int(x.size(1)/self.patch_dim),self.patch_dim)
        return x

    def forward(self,x):
        x=self.slice(x)
        x=self.embedding(x)

        return x

class multiattn(nn.Module):
    def __init__(self,d_model,nhead):
        super(multiattn, self).__init__()
        self.nhead = nhead
        self.proj_dim = int(d_model / nhead)
        self.q_proj_weight = nn.Linear(d_model, d_model)
        self.k_proj_weight = nn.Linear(d_model, d_model)
        self.v_proj_weight = nn.Linear(d_model, d_model)
        self.o_proj_weight = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        out = self._multiattn(q, k, v)
        return out

    def _multiattn(self, q, k, v):
        S_q=q.size(1)
        B, S, C = k.size()
        q_proj = self.q_proj_weight(q).reshape(B, S_q, self.nhead, self.proj_dim).permute(0, 2, 1, 3)
        k_proj = self.k_proj_weight(k).reshape(B, S, self.nhead, self.proj_dim).permute(0, 2, 1, 3)
        v_proj = self.v_proj_weight(v).reshape(B, S, self.nhead, self.proj_dim).permute(0, 2, 1, 3)
        attn_score = torch.matmul(q_proj, k_proj.transpose(-1, -2))
        attn_score = self.softmax(attn_score / math.sqrt(self.proj_dim))
        # ax = sns.heatmap(attn_score[1,0,:,:])
        # plt.show()

        attn_map = torch.matmul(attn_score, v_proj).permute(0, 2, 1, 3).reshape(B, S_q, C)
        attn_out = self.o_proj_weight(attn_map)
        return attn_out

class encoder_layer(nn.Module):
    def __init__(self,d_model,nhead,dim_feedforward):
        super(encoder_layer, self).__init__()

        self.self_aten=multiattn(d_model,nhead)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,q,k,v):
        out= q + self.self_aten(self.norm1(q),self.norm1(k),self.norm1(v))

        out = out + self.ffn(self.norm2(out))

        return out

class sparse_attn(nn.Module):
    def __init__(self, q_dim, sampling_nums):
        super().__init__()

        self.sampling_linear = nn.Linear(q_dim, sampling_nums)
        self.weight_linear = nn.Linear(q_dim, sampling_nums)
        self.value_linear = nn.Linear(q_dim, q_dim)
        self.output_linear = nn.Linear(q_dim, q_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        sampling_index = self.sampling_linear(x)
        sampling_index = (self.softmax(sampling_index) * len(sampling_index)).int()

        sampling_weight = self.weight_linear(x)
        sampling_weight = self.softmax(sampling_weight)

        values = self.value_linear(x)
        values_sampling = []
        for i, index in enumerate(sampling_index):
            value = values[index, :]
            values_sampling.append(torch.matmul(sampling_weight[i, :].unsqueeze(0), value.transpose(0, 1)))
        values_sampling = torch.cat(values_sampling, dim=0)

        output = self.output_linear(values_sampling)

        return output

class encoder_sparselayer(nn.Module):
    def __init__(self,d_model,sampling_num,dim_feedforward):
        super().__init__()

        self.self_aten=sparse_attn(d_model,sampling_num)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x):
        out= x + self.self_aten(self.norm1(x))

        out = out + self.ffn(self.norm2(out))

        return out

class model_transformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv1d(1,32,3,padding='same'),
            nn.AvgPool1d(2,2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(32,64,3,padding='same'),
            nn.AvgPool1d(2,2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(64,128,3,padding='same'),
            nn.AvgPool1d(2,2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding='same'),
            nn.AvgPool1d(2, 2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.position_embeding=PositionalEncoding(d_model=128)
        self.encoder1=encoder_layer(d_model=128,nhead=2,dim_feedforward=256)
        self.encoder2=encoder_layer(d_model=128,nhead=2,dim_feedforward=256)
        self.encoder3=encoder_layer(d_model=128,nhead=2,dim_feedforward=256)
        self.encoder4=encoder_layer(d_model=128,nhead=2,dim_feedforward=256)
        self.adaptiveavgpool=nn.AdaptiveAvgPool1d(1)

        self.mlp=nn.Sequential(
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32,1),
        )

    def forward(self,x):
        x=x.permute(0,2,1)

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        x=out_conv4.permute(0,2,1)

        x=self.position_embeding(x)
        x = self.encoder1(x, x, x)
        x = self.encoder2(x, x, x)
        x = self.encoder3(x, x, x)
        x = self.encoder4(x, x, x)

        x=x.permute(0,2,1)
        x=self.adaptiveavgpool(x)
        x=x.squeeze()

        x=self.mlp(x)
        return x

class model_transformer_no_posi(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv1d(1,32,3,padding='same'),
            nn.AvgPool1d(2,2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(32,64,3,padding='same'),
            nn.AvgPool1d(2,2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(64,128,3,padding='same'),
            nn.AvgPool1d(2,2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding='same'),
            nn.AvgPool1d(2, 2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.position_embeding=PositionalEncoding(d_model=128)
        self.encoder1=encoder_layer(d_model=128,nhead=2,dim_feedforward=256)
        self.encoder2=encoder_layer(d_model=128,nhead=2,dim_feedforward=256)
        self.encoder3=encoder_layer(d_model=128,nhead=2,dim_feedforward=256)
        self.encoder4=encoder_layer(d_model=128,nhead=2,dim_feedforward=256)
        self.adaptiveavgpool=nn.AdaptiveAvgPool1d(1)

        self.mlp=nn.Sequential(
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32,1),
        )

    def forward(self,x):
        x=x.permute(0,2,1)

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        x=out_conv4.permute(0,2,1)

        #x=self.position_embeding(x)
        x = self.encoder1(x, x, x)
        x = self.encoder2(x, x, x)
        x = self.encoder3(x, x, x)
        x = self.encoder4(x, x, x)

        x=x.permute(0,2,1)
        x=self.adaptiveavgpool(x)
        x=x.squeeze()

        x=self.mlp(x)
        return x

class single_conv(nn.Module):
    def __init__(self,in_ch,out_ch,ker_size=3,stride=1,padding=1):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(in_ch,out_ch,ker_size,stride=stride,padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x=self.conv(x)

        return x

class model_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layer1=nn.Sequential(
            nn.Linear(6000,170), #Strong nonlinear response is 225
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )
        self.fc_layer2=nn.Sequential(
            nn.Linear(170,128),
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )
        self.fc_layer4 = nn.Linear(64,1)


    def forward(self,x):
        x=x.squeeze()
        x=self.fc_layer1(x)
        x=self.fc_layer2(x)
        x=self.fc_layer3(x)
        x=self.fc_layer4(x)

        return x

class model_lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 300, 1, batch_first=True,dropout=0.3)
        self.lstm2 = nn.LSTM(300, 256, 1, batch_first=True,dropout=0.3)
        self.lstm3 = nn.LSTM(256, 64, 1, batch_first=True,dropout=0.3)
        self.lstm4 = nn.LSTM(64,1,1,batch_first=True,dropout=0.3)

    def forward(self,x):

        x,(h,c)=self.lstm1(x)
        x,(h,c)=self.lstm2(x)
        x,(h,c)=self.lstm3(x)
        x,(h,c)=self.lstm4(x)

        x=x[:,-1,:]

        return x

class model_single_path(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1=single_conv(1,32)
        self.conv2=single_conv(32,64)
        self.conv3=single_conv(64,128)
        self.conv4 =single_conv(128,128)
        self.avg_pooling=nn.AvgPool1d(2,2)

        self.lstm1=nn.LSTM(128,128,1,batch_first=True)
        self.lstm2=nn.LSTM(128,256,1,batch_first=True)
        self.lstm3=nn.LSTM(256,256,1,batch_first=True)
        self.lstm4=nn.LSTM(256,128,1,batch_first=True)

        self.mlp=nn.Sequential(
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32,1)
        )
    def forward(self,x):

        x=x.permute(0,2,1)
        out_conv1=self.conv1(x)
        out_conv1=self.avg_pooling(out_conv1)

        out_conv2=self.conv2(out_conv1)
        out_conv2=self.avg_pooling(out_conv2)

        out_conv3=self.conv3(out_conv2)
        out_conv3=self.avg_pooling(out_conv3)

        out_conv4 = self.conv4(out_conv3)
        out_conv4 = self.avg_pooling(out_conv4)

        out=out_conv4.permute(0,2,1)
        out,(h_n,c_n)=self.lstm1(out)
        out, (h_n, c_n) = self.lstm2(out)
        out, (h_n, c_n) = self.lstm3(out)
        out, (h_n, c_n) = self.lstm4(out)
        out=self.mlp(out[:,-1,:])

        return out

class model_dual_path(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.Embedding = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(128, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2, 2),
        )

        self.Temporal_prior = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2, 2),
        )

        self.norm1 = nn.LayerNorm(out_ch)
        self.norm3 = nn.LayerNorm(out_ch)
        self.injector1 = multiattn(d_model=out_ch, nhead=1)
        self.injector3 = multiattn(d_model=out_ch, nhead=1)

        self.gamma1 = nn.Parameter(torch.randn(1, out_ch))
        self.gamma3 = nn.Parameter(torch.randn(1, out_ch))  

        self.lstm1 = nn.LSTM(out_ch, out_ch, 1, batch_first=True)
        self.lstm2 = nn.LSTM(out_ch, out_ch, 1, batch_first=True)
        self.lstm3 = nn.LSTM(out_ch, out_ch, 1, batch_first=True)
        self.lstm4 = nn.LSTM(out_ch, out_ch, 1, batch_first=True)

        self.position_embeding = PositionalEncoding(d_model=out_ch)
        self.encoder1 = encoder_layer(d_model=out_ch, nhead=4, dim_feedforward=out_ch * 2)
        self.encoder3 = encoder_layer(d_model=out_ch, nhead=4, dim_feedforward=out_ch * 2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)

        self.Mlp = nn.Sequential(
            nn.Linear(out_ch, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 1),
        )

    def forward(self, x):

        x_equal = x[:, :, 0].unsqueeze(dim=2)
        x_variable = x[:, :, 1].unsqueeze(dim=2)

        x_embedding = self.Embedding(x_equal.permute(0, 2, 1)).permute(0, 2, 1)
        x_embedding = self.position_embeding(x_embedding)

        temporal_prior = self.Temporal_prior(x_variable.permute(0, 2, 1)).permute(0, 2, 1)
        temporal_prior = self.position_embeding(temporal_prior)

        injector1 = self.injector1(self.norm1(x_embedding), self.norm1(temporal_prior), self.norm1(temporal_prior))
        x_lstm1, (h_n, c_n) = self.lstm1(x_embedding + self.gamma1 * injector1)
        x_encoder1 = self.encoder1(temporal_prior, x_lstm1, x_lstm1)

        x_lstm2, (h_n, c_n) = self.lstm2(x_lstm1)

        injector3 = self.injector3(self.norm3(x_lstm2), self.norm3(x_encoder1), self.norm3(x_encoder1))
        x_lstm3, (h_n, c_n) = self.lstm3(x_lstm2 + self.gamma3 * injector3)
        x_encoder3 = self.encoder3(x_encoder1, x_lstm3, x_lstm3)

        x_lstm4, (h_n, c_n) = self.lstm4(x_lstm3)

        x = self.adaptiveavgpool(x_encoder3.permute(0, 2, 1)).squeeze()
        x = x + x_lstm4[:, -1, :]

        x = self.Mlp(x)

        return x

class model_dual_path_nonlinear(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.Embedding = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(128, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.Tanh(),
            nn.AvgPool1d(2, 2),
        )

        self.Temporal_prior = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.AvgPool1d(2, 2),
        )

        self.norm1 = nn.LayerNorm(out_ch)
        self.norm3 = nn.LayerNorm(out_ch)
        self.injector1 = multiattn(d_model=out_ch, nhead=1)
        self.injector3 = multiattn(d_model=out_ch, nhead=1)

        self.gamma1 = nn.Parameter(torch.randn(1, out_ch))
        self.gamma3 = nn.Parameter(torch.randn(1, out_ch))

        self.lstm1 = nn.LSTM(out_ch, out_ch, 1, batch_first=True)
        self.lstm2 = nn.LSTM(out_ch, out_ch, 1, batch_first=True)
        self.lstm3 = nn.LSTM(out_ch, out_ch, 1, batch_first=True)
        self.lstm4 = nn.LSTM(out_ch, out_ch, 1, batch_first=True)

        self.position_embeding = PositionalEncoding(d_model=out_ch)
        self.encoder1 = encoder_layer(d_model=out_ch, nhead=4, dim_feedforward=out_ch * 2)
        self.encoder3 = encoder_layer(d_model=out_ch, nhead=4, dim_feedforward=out_ch * 2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)

        self.Mlp = nn.Sequential(
            nn.Linear(out_ch, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),

            nn.Linear(32, 1),
        )


        self.fc1=nn.Linear(out_ch,64)
        self.fc1_bn=nn.BatchNorm1d(64)
        self.tanh=nn.Tanh()

        self.fc2=nn.Linear(64, 32)
        self.fc2_bn=nn.BatchNorm1d(32)

        self.fc3=nn.Linear(32, 1)


    def forward(self, x):

        x_equal = x
        x_variable = x

        x_embedding = self.Embedding(x_equal.permute(0, 2, 1)).permute(0, 2, 1)
        x_embedding = self.position_embeding(x_embedding)

        temporal_prior = self.Temporal_prior(x_variable.permute(0, 2, 1)).permute(0, 2, 1)
        temporal_prior = self.position_embeding(temporal_prior)

        injector1 = self.injector1(self.norm1(x_embedding), self.norm1(temporal_prior), self.norm1(temporal_prior))
        x_lstm1, (h_n, c_n) = self.lstm1(x_embedding + self.gamma1 * injector1)
        x_encoder1 = self.encoder1(temporal_prior, x_lstm1, x_lstm1)

        x_lstm2, (h_n, c_n) = self.lstm2(x_lstm1)

        injector3 = self.injector3(self.norm3(x_lstm2), self.norm3(x_encoder1), self.norm3(x_encoder1))
        x_lstm3, (h_n, c_n) = self.lstm3(x_lstm2 + self.gamma3 * injector3)
        x_encoder3 = self.encoder3(x_encoder1, x_lstm3, x_lstm3)

        x_lstm4, (h_n, c_n) = self.lstm4(x_lstm3)

        x = self.adaptiveavgpool(x_encoder3.permute(0, 2, 1)).squeeze()
        x = x + x_lstm4[:, -1, :]

        x=self.fc1(x)
        x=self.fc1_bn(x)
        x=self.tanh(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.tanh(x)

        x = self.fc3(x)

        return x

if __name__ == '__main__':
    m=model_dual_path(out_ch=128)
    params = sum(p.numel() for p in m.parameters())
    print(params)
