import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GAE_Loss(nn.Module):
    def __init__(self, lambda_reg):
        super(GAE_Loss, self).__init__()
        self.lambda_reg = lambda_reg

    def l2_regularization(self, model):# 对模型进行正则化，助于学习到更稳定和泛化的表示，防止过拟合

        l2_reg = torch.tensor(0.0,device=device) #初始化为0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.lambda_reg * l2_reg

    def recon_loss_a(self, adj, adj_pred):
        # 计算重构损失，使用交叉熵
        loss_restrct_a = F.mse_loss(adj_pred, adj)
        return loss_restrct_a

    def recon_loss_b(self, adj, adj_pred):
        # 计算重构损失
        loss_restrct_b = F.mse_loss(adj_pred, adj)
        return loss_restrct_b

    def kl_loss_a(self, mean, logstd):
        # 计算KL散度，使用闭合形式
        loss_kl_a = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mean.pow(2) - logstd.exp().pow(2), dim=1))
        return loss_kl_a

    def kl_loss_b(self, mean, logstd):
        # 计算KL散度，使用闭合形式
        loss_kl_b = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mean.pow(2) - logstd.exp().pow(2), dim=1))
        return loss_kl_b

    def con_loss_mean(self,mean_a,mean_b):
        con_loss_mean=F.mse_loss(mean_a,mean_b)
        return con_loss_mean

    def con_loss_wz(self, w, z):
        con_loss_wz = F.mse_loss(w, z)
        return con_loss_wz
    def con_loss_x(self,x_a,x_b):
        con_loss_x=F.mse_loss(x_a,x_b)

        return con_loss_x

    def con_loss_logist(self, log_a, log_b):
        con_loss_logist = F.mse_loss(log_a,log_b)
        return con_loss_logist



    def forward(self, node_t1, adj_a, node_t2, adj_b,mean_a,logstd_a,mean_b,logstd_b,encoder, x_a, x_b,decoder_a,decoder_b):#函数调用
        recon_loss_a = self.recon_loss_a(adj_a,node_t1)
        recon_loss_b = self.recon_loss_b(adj_b,node_t2)
        con_loss_x = self.con_loss_x(x_a, x_b)
        con_loss_mean=self.con_loss_mean(mean_a,mean_b)
        con_loss_logist=self.con_loss_logist(logstd_a,logstd_b)
        loss_kl_a=self.kl_loss_a(mean_a,logstd_a)
        loss_kl_b = self.kl_loss_b(mean_b,logstd_b)
        l2reg_loss = self.l2_regularization(encoder)
        l2reg_loss_da = self.l2_regularization(decoder_a)
        l2reg_loss_db = self.l2_regularization(decoder_b)
        return recon_loss_a + recon_loss_b +l2reg_loss_db+l2reg_loss_da+l2reg_loss+con_loss_x+con_loss_logist+con_loss_mean+loss_kl_a+loss_kl_b
