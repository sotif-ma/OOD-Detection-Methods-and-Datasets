import torch
import torch.nn.functional as F

from .recorder import Recorder

def msp(recorder:Recorder, temp=1.0):
  logit = recorder._recorders['_logit'].compute()
  return F.softmax(logit/temp, dim=-1).max(dim=-1)[0]

def energy(recorder:Recorder):
  logit = recorder._recorders['_logit'].compute()
  return torch.logsumexp(logit, -1)

def trim(recorder:Recorder, start:int=5, end:int=15):
  logit = recorder._recorders['_logit'].compute()
  sorted_prob = F.softmax(logit, dim=-1).sort(dim=-1, descending=True)[0]
  return (1-sorted_prob[:,start:end]).sum(dim=-1)

