import numpy as np
import torch
import torch.nn.functional as F
from .runner import OODRunner

def dice(runner:OODRunner, linear_layer:torch.nn.Linear, percentile:float=90.0) -> OODRunner:
  feature = runner.id_recorder.feature.compute().flatten(start_dim=1)
  avg = feature.mean(dim=0)
  weight = linear_layer.weight.detach().cpu()
  bias = linear_layer.bias
  if bias is None:
    bias = torch.zeros((weight.shape[0]))
  else:
    bias = bias.detach().cpu()

  dice_runner = runner.clone()
  with torch.no_grad():
    contrib = avg * weight
    thresh = np.percentile(contrib, percentile)
    masked_weight = (contrib > thresh) * weight

    dice_runner.id_recorder._logit.value = [F.linear(x.flatten(start_dim=1), masked_weight, bias) for x in runner.id_recorder.feature.value]
    for k, recorder in dice_runner.ood_recorders.items():
      recorder._logit.value = [F.linear(x.flatten(start_dim=1), masked_weight, bias) for x in runner.ood_recorders[k].feature.value]
  return dice_runner

def _ash_s(x:torch.Tensor, percentile:float=90.0):
    assert (x.dim() == 4) or (x.dim() == 2)
    assert 0 <= percentile <= 100
    
    x = x.clone()
    if x.dim() == 2:
        x = x.reshape(*x.shape, 1, 1)
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x

def ash_s(runner:OODRunner, linear_layer:torch.nn.Linear, percentile:float=90.0) -> OODRunner:
    weight = linear_layer.weight.detach().cpu()
    bias = linear_layer.bias
    if bias is None:
        bias = torch.zeros((weight.shape[0]))
    else:
        bias = bias.detach().cpu()

    ash_runner = runner.clone()
    with torch.no_grad():
        ash_runner.id_recorder._logit.value =  [F.linear(_ash_s(x, percentile).flatten(start_dim=1), weight, bias) for x in runner.id_recorder.feature.value]
        for k, recorder in ash_runner.ood_recorders.items():
            recorder._logit.value = [F.linear(_ash_s(x).flatten(start_dim=1), weight, bias) for x in runner.ood_recorders[k].feature.value]
    return ash_runner