import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import os
import json

def calc_recalls(image_outputs, audio_outputs, args, nframes, simtype='MISA', nregions=None):
    """
	Computes recall at 1, 5, and 10 given encoded image and audio outputs.
	"""
    if args.feature == 'vector':
        image_outputs = image_outputs.unsqueeze(0)
        audio_outputs = audio_outputs.unsqueeze(0)
        
        image_outputs_norm = torch.norm(image_outputs, 2, dim=2, keepdim=True)
        audio_outputs_norm = torch.norm(audio_outputs, 2, dim=2, keepdim=True)
        S = torch.bmm(image_outputs, audio_outputs.transpose(1, 2))
        S = S.squeeze()
        if args.alignment_scores:
            S_a = torch.FloatTensor(np.load(args.alignment_scores))
            S = (S.softmax(0) + S.softmax(1)) / 2 * S_a
    else:
        if simtype == 'ASISA' or simtype == 'BASISA':
            print('Compute attentive matchmap similarity ...') # XXX
            S = compute_attentive_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, nregions=nregions, simtype=simtype)
        else:
            S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, nregions=nregions, simtype=simtype)
        np.save('{}/similarity.npy'.format(args.exp_dir), S.cpu().detach().numpy())
        
        if args.alignment_scores:
            S_a = torch.FloatTensor(np.load(args.alignment_scores))
            S = (S.softmax(0) + S.softmax(1)) / 2 * S_a # XXX
            
    n = S.size(0)
    # pdb.set_trace()
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}
    save_results_A2I = os.path.join(args.exp_dir,'A2I.text')
    save_results_I2A = os.path.join(args.exp_dir,'I2A.text')
    np.savetxt(save_results_A2I,A2I_ind.transpose(1,0).int().numpy(),fmt='%d')
    np.savetxt(save_results_I2A,I2A_ind.int().numpy(),fmt='%d')
    return recalls

def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)  
    return matchmap

def computeAttentiveSim(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = (matchmap / np.sqrt(D)).softmax(-1) * matchmap  
    return matchmap.sum(-1).mean()

def computeBiAttentiveSim(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap_A2I = (matchmap / np.sqrt(D)).softmax(-1) * matchmap  
    matchmap_I2A = (matchmap / np.sqrt(D)).softmax(0) * matchmap
    return (matchmap_A2I.sum(-1).mean() + matchmap_I2A.sum(0).mean()) / 2.
    
    
def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA', nregions=None):
    """
    image_outputs: B x D x H x W tensor
    audio_outputs: B x D x T  tensor
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    if n == 1:
      return loss
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        if len(nregions):
          nR = nregions[i]
          nRimp = nregions[I_imp_ind]
          anchorsim = matchmapSim(computeMatchmap(image_outputs[i][:, 0:nR], audio_outputs[i][:, 0:nF]), simtype)
          Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind][:, 0:nRimp], audio_outputs[i][:, 0:nF]), simtype)
          Aimpsim = matchmapSim(computeMatchmap(image_outputs[i][:, 0:nR], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        else:
          anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
          Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
          Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def mask_margin_softmax_loss(image_outputs, audio_outputs, nframes, margin=0.001, simtype='MISA', nregions=None):
    """
    image_outputs: B x D x H x W tensor
    audio_outputs: B x D x T  tensor
    Computes the masked margin softmax loss for each anchor image/caption pair as in:
    G. Ilharco, Y. Zhang, J. Baldridge. ``Large-scale representation learning from visually grounded untranscribed speech``. CoNLL, 2019.
    """
    # loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype, nregions=nregions)
    m = nn.LogSoftmax(dim=1) 
    n = image_outputs.size(0)
    loss = -torch.sum(m(S).diag())-torch.sum(m(S.transpose(0, 1)).diag())
    loss = loss / n
    return loss

def attentive_mask_margin_softmax_loss(image_outputs, audio_outputs, attention_model, nframes, margin=0.001, simtype='ASISA', nregions=None):
    """
    image_outputs: B x D x R tensor
    audio_outputs: B x D x T  tensor
    Computes the masked margin softmax loss for each anchor image/caption pair as in:
    G. Ilharco, Y. Zhang, J. Baldridge. ``Large-scale representation learning from visually grounded untranscribed speech``. CoNLL, 2019.
    """
    # loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    # S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, attention_model, nframes, simtype=simtype, nregions=nregions)
    m = nn.LogSoftmax(dim=1) 
    n = image_outputs.size(0)
    S = compute_attentive_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype, nregions=nregions)
    loss = -torch.sum(m(S).diag())-torch.sum(m(S.transpose(0, 1)).diag())
    loss = loss / n
    return loss

def DAMSM_loss(image_outputs, audio_outputs, nframes, margin=0.001, simtype='MISA', nregions=None):
    """
    image_outputs: B x D x H x W tensor
    audio_outputs: B x D x T  tensor
    Tao Xu. 2018. CVPR. Fine-Grained Text to Image Generation with Attention Generative Adversarial Networks
    """
    batch_size = image_outputs.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).cuda()
    scores0 = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype, nregions=nregions)*10.0
    scores1 = scores0.transpose(0, 1)
    loss = nn.CrossEntropyLoss()(scores0, labels) + nn.CrossEntropyLoss()(scores1, labels)
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA', nregions=None):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                if len(nregions):
                  nR = max(1, nregions[image_idx])
                  S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx][:, 0:nR], audio_outputs[audio_idx][:, 0:nF]), simtype)
                else:
                  S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx][:, 0:nR], audio_outputs[audio_idx][:, 0:nF]), simtype)
                '''
                if image_idx == 0 and audio_idx == 0:
                    Mmap = computeMatchmap(image_outputs[image_idx][:, 0:nR], audio_outputs[audio_idx][:, 0:nF]).mean(-1).cpu().detach().numpy()
                    print('Mmap.shape: {}'.format(Mmap.shape))
                    with open('matchmap.json', 'w') as f:
                        json.dump(Mmap.tolist(), f)
                '''
    return S

def compute_attentive_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='ASISA', nregions=None):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                if len(nregions):
                  nR = max(1, nregions[image_idx])
                  if simtype == 'ASISA':
                      S[image_idx, audio_idx] = computeAttentiveSim(image_outputs[image_idx][:, 0:nR], audio_outputs[audio_idx][:, 0:nF])
                  elif simtype == 'BASISA':
                      S[image_idx, audio_idx] = computeBiAttentiveSim(image_outputs[image_idx][:, 0:nR], audio_outputs[audio_idx][:, 0:nF])
                else:
                  if simtype == 'ASISA':
                      S[image_idx, audio_idx] = computeAttentiveSim(image_outputs[image_idx][:, 0:nR], audio_outputs[audio_idx][:, 0:nF])
                  elif simtype == 'BASISA':
                      S[image_idx, audio_idx] = computeBiAttentiveSim(image_outputs[image_idx][:, 0:nR], audio_outputs[audio_idx][:, 0:nF])
    return S


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

def vector_sim(x,y):
    sim = torch.mm(x.unsqueeze(0),y.unsqueeze(-1))
    return sim


def random_negative_mining_pair(image_output,audio_output):
    n = image_output.shape[0]
    I_imp_inds = []
    A_imp_inds = []
    for i in range(n):
        I_imp_ind = np.random.randint(0, n)
        A_imp_ind = np.random.randint(0, n)
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
    
        I_imp_inds.append(I_imp_ind)
        A_imp_inds.append(A_imp_ind)

    neg_audio = audio_output[A_imp_inds]
    neg_img = image_output[I_imp_inds]
    return neg_audio, neg_img


def sampled_margin_rank_loss_vector(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA', nregions=None):
    """
    image_outputs: B x D vector
    audio_outputs: B x D tensor
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
     
        anchorsim = vector_sim(image_outputs[i], audio_outputs[i])
        Iimpsim = vector_sim(image_outputs[I_imp_ind], audio_outputs[i])
        Aimpsim = vector_sim(image_outputs[i], audio_outputs[A_imp_ind])
        
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def sampled_margin_rank_loss_vector_opt(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA', nregions=None):
    """
    image_outputs: B x D vector
    audio_outputs: B x D tensor
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    n = image_outputs.size(0)
    neg_audio, neg_img = random_negative_mining_pair(image_outputs,audio_outputs) 
    anchorsim = torch.bmm(image_outputs.unsqueeze(1),audio_outputs.unsqueeze(-1))
    Iimpsim = torch.bmm(neg_img.unsqueeze(1),audio_outputs.unsqueeze(-1))
    Aimpsim = torch.bmm(image_outputs.unsqueeze(1),neg_audio.unsqueeze(-1))
    loss = (nn.ReLU()(margin + Aimpsim - anchorsim)).mean() + (nn.ReLU()(margin + Iimpsim - anchorsim)).mean()

    
    return loss



def mask_margin_softmax_loss_vector(image_outputs, audio_outputs, nframes, margin=0.001, simtype='MISA', nregions=None):
    """
    image_outputs: B x D x H x W tensor
    audio_outputs: B x D x T  tensor
    Computes the masked margin softmax loss for each anchor image/caption pair as in:
    G. Ilharco, Y. Zhang, J. Baldridge. ``Large-scale representation learning from visually grounded untranscribed speech``. CoNLL, 2019.
    """
    # loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    batch_size = image_outputs.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).cuda()
    # image_outputs = image_outputs.unsqueeze(0)
    # audio_outputs = audio_outputs.unsqueeze(0)
    # pdb.set_trace()
    S = torch.mm(image_outputs, audio_outputs.transpose(1, 0))
    # S = scores.squeeze()
    m = nn.LogSoftmax() 
    n = image_outputs.size(0)
    loss = -torch.sum(m(S).diag())-torch.sum(m(S.T).diag())
    loss = loss / n
    return loss


def DAMSM_loss_vector(image_outputs, audio_outputs, nframes, margin=0.001, simtype='MISA', nregions=None):
    """
    image_outputs: B x D x H x W tensor
    audio_outputs: B x D x T  tensor
    Computes the masked margin softmax loss for each anchor image/caption pair as in:
    G. Ilharco, Y. Zhang, J. Baldridge. ``Large-scale representation learning from visually grounded untranscribed speech``. CoNLL, 2019.
    """
    # loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    batch_size = image_outputs.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).cuda()
    image_outputs = image_outputs.unsqueeze(0)
    audio_outputs = audio_outputs.unsqueeze(0)
    
    image_outputs_norm = torch.norm(image_outputs, 2, dim=2, keepdim=True)
    audio_outputs_norm = torch.norm(audio_outputs, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(image_outputs, audio_outputs.transpose(1, 2))
    norm0 = torch.bmm(image_outputs_norm, audio_outputs_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=1e-5) * 10
    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)
    loss = nn.CrossEntropyLoss()(scores0, labels) + nn.CrossEntropyLoss()(scores1, labels)
    
    return loss
