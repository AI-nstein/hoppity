import torch
import random
import numpy as np
from torchext import jagged_argmax, jagged2padded, nn_distance, approx_match, match_cost, jagged_log_softmax, jagged_topk, replace_rows, jagged_append
from torch.optim import SGD
from torch.nn import Parameter
from torch import nn

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

def test_jagged_argmax():
    v = torch.rand(10, dtype=torch.float64)
    print(v)
    psum = torch.LongTensor([2, 5, 10])
    out = jagged_argmax(v, psum)
    print(out)

    v = v.to(DEVICE)
    psum = psum.to(DEVICE)
    out = jagged_argmax(v, psum)
    print(out)

def test_jagged_append():
    class TestCls(nn.Module):
        def __init__(self):
            super(TestCls, self).__init__()
            self.v = Parameter(torch.randn(10, ))
            print(self.v)
            self.suffix = Parameter(torch.randn(5, 2))
            print(self.suffix)
    psum = torch.LongTensor([2, 5, 6, 6, 10])
    tt = TestCls()
    mask = None
    for dev in ['cpu', 'gpu']:
        print(dev)
        if dev == 'gpu':
            tt = tt.cuda()
            psum = psum.cuda()
        v = tt.v
        suffix = tt.suffix
        v.grad = None
        suffix.grad = None

        out = jagged_append(v, psum, suffix)
        print(out)

        ## test grad
        if mask is None:
            mask = torch.randn(out.shape)
            print('mask')
            print(mask)
        if dev == 'gpu':
            mask = mask.cuda()
        t = torch.sum(mask * out)
        t.backward()

        print(v.grad)
        print(suffix.grad)

def test_jagged2padded():
    v = torch.rand(10, dtype=torch.float64)
    print(v)
    psum = torch.LongTensor([2, 5, 6, 6, 10])
    out = jagged2padded(v, psum, np.pi)
    print(out)

    v = v.to(DEVICE)
    psum = psum.to(DEVICE)
    out = jagged2padded(v, psum, np.pi)
    print(out)

def test_jagged_topk():
    v = torch.rand(10, dtype=torch.float64)
    print(v)
    psum = torch.LongTensor([2, 5, 6, 6, 10])
    v = v.to(DEVICE)
    psum = psum.to(DEVICE)    
    tv, tk = jagged_topk(v, psum, 2)    
    print('largest:')
    print(tv)
    print(tk)
    tv, tk = jagged_topk(v, psum, 2, largest=False)
    print('smallest:')
    print(tv)
    print(tk)

def test_jagged_log_softmax():
    v = torch.rand(10, dtype=torch.float32)
    v.requires_grad = True
    print('v', v)
    psum = torch.LongTensor([2, 5, 10])
    out = jagged_log_softmax(v, psum)
    loss = torch.sum(out)
    loss.backward()
    print('out', torch.exp(out))
    print('grad', v.grad)

    print('====testing gpu====')    
    v = torch.tensor(v.data.numpy(), dtype=torch.float32).cuda()
    v.requires_grad = True
    psum = psum.cuda()
    out = jagged_log_softmax(v, psum)
    loss = torch.sum(out)
    loss.backward()
    print('out', torch.exp(out))
    print('grad', v.grad)

def test_nn_dist():
    np.random.seed(100)
    xyz1=np.random.randn(32,16,3).astype('float32')
    xyz2=np.random.randn(32,10,3).astype('float32')

    inp1 = torch.tensor(xyz1).to(DEVICE)
    inp1.requires_grad = True
    inp2 = torch.tensor(xyz2).to(DEVICE)

    optimizer = SGD([inp1], lr=0.05)
    for i in range(100):
        optimizer.zero_grad()
        reta,retb,retc,retd = nn_distance(inp1,inp2)
        loss=torch.sum(reta)+torch.sum(retc)
        loss.backward()
        optimizer.step()
        print(i,loss.item())


def test_emd():
    random.seed(1)
    np.random.seed(1)

    tpoints=np.hstack([np.linspace(-1,1,400)[:,None],(random.random()*2*np.linspace(1,0,400)**2)[:,None],np.zeros((400,1))])[None,:,:]
    pt_in = torch.tensor(tpoints.astype('float32')).to(DEVICE)
    npoint=100
    mypoints=torch.tensor(np.random.randn(1,npoint,3).astype('float32')).to(DEVICE)
    mypoints.requires_grad = True
    optimizer = SGD([mypoints], lr=1e-4)

    for i in range(1000):
        optimizer.zero_grad()
        match=approx_match(pt_in,mypoints)
        loss=torch.sum(match_cost(pt_in,mypoints,match))
        print(i, loss.item())

        loss.backward()
        optimizer.step()


def test_replace_rows():
    a = torch.randn(5, 2)
    b = torch.randn(3, 2)

    a.requires_grad = True
    b.requires_grad = True
    indices = torch.LongTensor([0, 2, 4])
    print('---a---')
    print(a)
    print('---b---')
    print(b)
    print('---c---')
    c = replace_rows(a, indices, b)
    print(c)

    t = torch.sum(c[0])
    t.backward()
    print('---ga---')
    print(a.grad)
    print('---gb---')
    print(b.grad)
    print('---gc---')
    print(c.grad)



if __name__ == '__main__':
    # test_jagged_argmax()
    # test_nn_dist()
    # test_emd()
    # test_jagged_log_softmax()
    # test_jagged_topk()
    # test_replace_rows()
    test_jagged_append()
