import torch.nn as nn
import numpy
import torch
import copy
from torch.autograd import Variable
from torch.nn.parameter import Parameter
class TerOp():
    def __init__(self, model, args):
        # count the number of Conv2d

        count_Conv2d = 0
        for n,m in model.named_modules():
            if isinstance(m, nn.Conv2d) and 'res' not in n and 'preprocess' not in n:
                count_Conv2d = count_Conv2d + 1

        start_range = 1
        end_range = count_Conv2d
        self.ter_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
     
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        self.num_skip = args.num_skip
        index = 0
        for n,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.ter_range and 'res' not in n and index >= (self.num_skip + 1) and 'preprocess' not in n:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
        self.num_of_params = len(self.target_modules)
        self.delta_w = Parameter(torch.ones(1,requires_grad=True, device="cuda")*0.7)
        self.initial_scaling_factors = []
    def initial_scailing_param(self):
        for index in range(self.num_of_params):
            w_p_initial, w_n_initial = self.initial_scales(self.target_modules[index].data)
            self.initial_scaling_factors += Variable(torch.FloatTensor([w_p_initial, w_n_initial]).cuda(), requires_grad=True)
    def initial_scales(kernel):
        return 1.0, 1.0
    def ternarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.terneraizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def terneraizeConvParams(self):
        for index in range(self.num_of_params):
            delta = self.Delta(self.target_modules[index].data)
            self.target_modules[index].data = self.sign_new(self.target_modules[index].data,delta)
    # def Ternarize(self,tensor):
    #     output = torch.zeros(tensor.size())
    #     output = output.type(torch.cuda.FloatTensor)
    #     delta = self.Delta(tensor)
    #     # print(delta.shape)
    #     alpha = self.Alpha(tensor,delta)
    #     # print(alpha.shape)
    #     for i in range(tensor.size()[0]):
    #         for w in tensor[i].view(1,-1):
    #             pos_one = (w > delta[i]).type(torch.cuda.FloatTensor)
    #             neg_one = torch.mul((w < -delta[i]).type(torch.cuda.FloatTensor),-1)
    #         out = torch.add(pos_one,neg_one).view(tensor.size()[1:])
    #         output[i] = torch.add(output[i],torch.mul(out,alpha[i]))
    #     return output#.cuda()###
    def sign_new(self,tensor,delta):
         output = torch.zeros_like(tensor)
         output = output.type(torch.cuda.FloatTensor)
         for i in range(tensor.size()[0]):
             output[i] = torch.where(torch.logical_and(tensor[i] >= -delta[i], tensor[i] <= delta[i]),
                                     torch.tensor(0.0).cuda(), self.initial_scaling_factors[i][0])
             output[i] = torch.where(tensor[i]<-delta,self.initial_scaling_factors[i][1],output[i])
         return output.cuda()
    def Delta(self,tensor):
        n = tensor[0].nelement() 
        s = tensor.size()
        if len(s) == 4: # binconv layer
            delta = self.delta_w * tensor.norm(1,3).sum(2).sum(1).div(n)
        elif len(s) == 2:
            delta = self.delta_w * tensor.norm(1,1).div(n)
        return delta
    # def Alpha(self,tensor,delta):
    #     Alpha = []
    #     for i in range(tensor.size()[0]):
    #         count = 0
    #         abssum = 0
    #         absvalue = tensor[i].view(1,-1).abs()
    #         for w in absvalue:
    #             truth_value = w > delta[i]
    #         count = truth_value.sum()
    #         count = count.type(torch.cuda.FloatTensor)
    #         abssum = torch.matmul(absvalue,truth_value.type(torch.cuda.FloatTensor).view(-1,1))
    #         Alpha.append(abssum/count)
    #     alpha = Alpha[0]
    #     for i in range(len(Alpha) - 1):
    #         alpha = torch.cat((alpha,Alpha[i+1]))
    #     return alpha##
    # def sign_new(self,tensor,delta,alpha):

    # def binarizeConvParams(self):
    #     for index in range(self.num_of_params):
    #         n = self.target_modules[index].data[0].nelement()
    #         s = self.target_modules[index].data.size()
    #         m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
    #                 .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
    #         self.target_modules[index].data = \
    #                 self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
    def sign_new(self,tensor, delta, alpha):
        tensor = tensor.cuda()
        delta = delta.cuda()
        alpha = alpha.cuda()
        output = torch.zeros_like(tensor)
        output = output.type(torch.cuda.FloatTensor)
        for i in range(tensor.size()[0]):
            output[i] = torch.where(torch.logical_and(tensor[i] >= -delta[i], tensor[i] <= delta[i]), torch.tensor(0.0).cuda(), alpha[i])
            output[i] = torch.where(tensor[i] < -delta[i], -alpha[i], output[i])
        return output.cuda()
    def updateTernaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            delta = self.Delta(weight)
            for i in range(weight.size()[0]):
                a = (weight[i] > delta[i]).float()
                b = (weight[i] < -delta[i]).float()
                c = torch.ones(weight[i].size()).cuda() - a - b
                self.target_modules[index][i].grad.data =\
                    self.initial_scaling_factors[index][0] * a * self.target_modules[index][i].grad.data\
                    + self.initial_scaling_factors[index][1] * b * self.target_modules[index][i].grad.data\
                    +1.0 * c * self.target_modules[index][i].grad.data
                self.initial_scaling_factors[index][0].grad.data = self.initial_scaling_factors[index][0].grad.data\
                    +(a * self.target_modules[index][i].grad.data).sum()
                self.initial_scaling_factors[index][1].grad.data = self.initial_scaling_factors[index][1].grad.data\
                   + (b * self.target_modules[index][i].grad.data).sum()
            self.initial_scaling_factors[index][0].grad.data =self.initial_scaling_factors[index][0].grad.data/weight.size()[0]
            self.initial_scaling_factors[index][1].grad.data = self.initial_scaling_factors[index][1].grad.data / weight.size()[0]
            A = weight.norm(1, 3, keepdim=False).sum(2, keepdim=False).sum(1, keepdim=False).div(self.delta_w)
            B = weight.grad.data.norm(1, 3, keepdim=False).sum(2, keepdim=False).sum(1,keepdim=False).div(n)
            self.delta_w.grad = self.delta_w.grad + torch.ones(1,requires_grad=True,device="cuda")*torch.mean(A * B)
        self.delta_w.grad = self.delta_w.grad  / self.num_of_params
        # def updateTernaryGradWeight(self):
    #     for index in range(self.num_of_params):
    #         weight = self.target_modules[index].data
    #         n = weight[0].nelement() # 500 # 800
    #         s = weight.size()#(50, 20, 5, 5) #  (500, 800)
    #         if len(s) == 4:
    #             delgrad = 0.7 * weight.norm(1, 3)\
    #                     .sum(2).sum(1).div(n)#.expand(s)
    #         elif len(s) == 2:
    #             delgrad = 0.7 * weight.norm(1, 1).div(n)#.expand(s)
        
    #         Alpha = []
    #         for i in range(weight.size()[0]):
    #             count = 0
    #             abssum = 0
    #             absvalue = weight[i].view(1,-1).abs()
    #             for w in absvalue:
    #                 truth_value = w > delgrad[i] #print to see
    #             count = truth_value.sum()
    #             count = count.type(torch.cuda.FloatTensor)
    #             abssum = torch.matmul(absvalue,truth_value.type(torch.cuda.FloatTensor).view(-1,1))
    #             Alpha.append(abssum/count)
    #         alpha = Alpha[0]
    #         for i in range(len(Alpha) - 1):
    #             alpha = torch.cat((alpha,Alpha[i+1]))
    #         if len(s) == 4:
    #             alpha1 = alpha[:,:,None,None]
    #             alpha1 = alpha1.expand(s)
    #         elif len(s) == 2:
    #             alpha1 = alpha.expand(s)
            
    #         alpha1[weight.lt(-1.0)] = 0 
    #         alpha1[weight.gt(1.0)] = 0  #m = alpha
    #         alpha1 = alpha1.mul(self.target_modules[index].grad.data)#alpha avaz shod

    #         #weight = weight.cpu()
	#     #output = torch.zeros(weight.size())
    #         outweight = torch.zeros(weight.size())
    #         outweight = outweight.type (torch.cuda.FloatTensor)
    #         for i in range(weight.size()[0]):
    #             for w in weight[i].view(1,-1):
    #                 pos_one = (w > delgrad[i]).type(torch.cuda.FloatTensor)
    #                 neg_one = torch.mul((w < - delgrad[i]).type(torch.cuda.FloatTensor),-1)
    #             out = torch.add(pos_one,neg_one).view(weight.size()[1:])
    #             outweight[i] = torch.add(outweight[i],out)
    #         #outweight = weight.sign()
   
    #         m_add = outweight.mul(self.target_modules[index].grad.data)
    #         if len(s) == 4:
    #             m_add = m_add.sum(3, keepdim=True)\
    #                 .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
    #         elif len(s) == 2:
    #             m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
    #         m_add = m_add.mul(outweight)
    #         self.target_modules[index].grad.data = alpha1.add(m_add).mul(1.0-1.0/s[1]).mul(n)
    #         self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(1e+9)