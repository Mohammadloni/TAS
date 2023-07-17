import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'bin_dil_conv_3x3' : lambda C, stride, affine: BinDilConv(C, C, 3, stride, 2, 2, affine=affine),
  'bin_dil_conv_5x5' : lambda C, stride, affine: BinDilConv(C, C, 5, stride, 4, 2, affine=affine),
  'bin_conv_3x3' : lambda C, stride, affine: BinReLUConvBN(C, C, 3, stride, 1, affine=affine),
  'bin_conv_5x5' : lambda C, stride, affine: BinReLUConvBN(C, C, 5, stride, 2, affine=affine) 
}


def sign_new(tensor, delta, alpha):
        tensor = tensor.cuda()
        delta = delta.cuda()
        alpha = alpha.cuda()
        output = torch.zeros_like(tensor)
        output = output.type(torch.cuda.FloatTensor)
        for i in range(tensor.size()[0]):
            output[i] = torch.where(torch.logical_and(tensor[i] >= -delta[i], tensor[i] <= delta[i]), torch.tensor(0.0).cuda(), alpha[i])
            output[i] = torch.where(tensor[i] < -delta[i], -alpha[i], output[i])
        return output.cuda()
def Delta(tensor,delta_I):
    n = tensor[0].nelement()
    s = tensor.size()
    if len(s) == 4:  # binconv layer
        delta = delta_I * tensor.norm(1, 3).sum(2).sum(1).div(n)
    elif len(s) == 2:
        delta = delta_I * tensor.norm(1, 1).div(n)
    return delta


def Alpha(tensor, delta):
    Alpha = []
    for i in range(tensor.size()[0]):
        count = 0
        abssum = 0
        absvalue = tensor[i].view(1, -1).abs()
        for w in absvalue:
            truth_value = w > delta[i]
        count = truth_value.sum()
        count = count.type(torch.cuda.FloatTensor)
        abssum = torch.matmul(absvalue, truth_value.type(torch.cuda.FloatTensor).view(-1, 1))
        Alpha.append(abssum / count)
    alpha = Alpha[0]
    for i in range(len(Alpha) - 1):
        alpha = torch.cat((alpha, Alpha[i + 1]))
    return alpha  ##

def compute_grad_input(input,grad_input,delta_I):
    for i in range(input.size()[0]):
        data = input[i].data
        # print(data.shape)
        n = data[0].nelement()
        A = data.norm(1, 2, keepdim=False).sum(1, keepdim=False).div(delta_I)
        B = grad_input[i].norm(1, 2, keepdim=False).sum(1, keepdim=False).div(n)
        return  torch.ones(1, requires_grad=True, device="cuda") * torch.mean(A * B)

class TerActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input,delta_I):
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        delta = Delta(input,delta_I)
        alpha = Alpha(input,delta)
        self.save_for_backward(input,delta,alpha,delta_I)
        input = sign_new(input,delta,alpha)
        return input, mean
    @staticmethod
    def backward(self, grad_output, grad_output_mean):
        input,delta,alpha ,delta_I= self.saved_tensors
        grad_input = grad_output.clone()
        for i in range(input.size()[0]):
          grad_input[input.ge(delta[i])] = 0
          grad_input[input.le(-delta[i])] = 0
        grad_delta_I = compute_grad_input(input,grad_input,delta_I)
        return grad_input,grad_delta_I

class BinReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(BinReLUConvBN, self).__init__()
    self.delta_I = Parameter(torch.ones(1,requires_grad=True, device="cuda")*0.7)
    self.bn = nn.BatchNorm2d(C_in, affine=affine)
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.ReLU(inplace=False),     
    )

  def forward(self, x):
     
    x = self.bn(x)
    x,_ = TerActive.apply(x,self.delta_I)
    return self.op(x)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.bn = nn.BatchNorm2d(C_in, affine=affine)
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.ReLU(inplace=False),     
    )

  def forward(self, x):  
    x = self.bn(x)  
    return self.op(x)


class BinDilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(BinDilConv, self).__init__()
    self.delta_I = Parameter(torch.ones(1, requires_grad=True, device="cuda") * 0.7)
    self.bn =  nn.BatchNorm2d(C_in, affine=affine)
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
       nn.ReLU(inplace=False),
      )

  def forward(self, x):
    x = self.bn(x)
    x, _ = TerActive.apply(x,self.delta_I)
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


