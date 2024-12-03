from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification.precision_recall_curve import _binary_precision_recall_curve_compute
from torchmetrics.functional.classification.roc import _binary_roc_compute

import pandas as pd
import matplotlib.pyplot as plt

from .runner import OODRunner, Recorder
from .scores import msp


def metrics(id_score, ood_score):
  score = torch.concat([id_score, ood_score])
  is_id = torch.concat([torch.ones_like(id_score), torch.zeros_like(ood_score)])

  fpr, tpr, _ = _binary_roc_compute((score, is_id), None)
  p_in, r_in, _ = _binary_precision_recall_curve_compute((score, is_id), None)
  p_out, r_out, _ = _binary_precision_recall_curve_compute((1-score, 1-is_id), None)

  return {
    'FPR@95': fpr[tpr <= 0.95].max().item(),
    'DTErr': ((1-tpr+fpr)/2).min().item(),
    'AUROC': torch.trapz(tpr, fpr).item(),
    'AUPR_In': -torch.trapz(p_in, r_in).item(),
    'AUPR_Out': -torch.trapz(p_out, r_out).item(),
  }


def metrics_df(runner:OODRunner, score_ftn:Callable) -> pd.DataFrame:
  assert runner._ran  
  id_score = score_ftn(runner.id_recorder)
  ood_scores = {k: score_ftn(v) for k,v in runner.ood_recorders.items()}
  
  return pd.DataFrame({
    k: metrics(id_score, v) for k,v in ood_scores.items()
  }).T

def hist(runner:OODRunner, ood:str, id:bool=True,
         score_ftn:Callable=msp,
         bins=500, range:Optional[tuple[float, float]]=None, alpha=0.5, id_name='', ood_name='') -> None:
  ood_recorder = runner.ood_recorders[ood]
  ood_score = score_ftn(ood_recorder)
  if id:
    id_recorder = runner.id_recorder
    id_score = score_ftn(id_recorder)
    if range is None:
      score = torch.concat([id_score, ood_score])
      range = (score.min().item(), score.max().item())
    plt.hist(id_score, bins=bins, range=range, alpha=alpha, label=f'{id_name} ({len(id_score)}, in-dist.)')
  plt.hist(ood_score, bins=bins, range=range, alpha=alpha, label=f'{ood_name} ({len(ood_score)}, out-of-dist.)')

def roc(runner:OODRunner, ood:str,
        score_ftn:Callable=msp,
        label_prefix='', **kwargs) -> float:
  id_recorder = runner.id_recorder
  ood_recorder = runner.ood_recorders[ood]
  id_score = score_ftn(id_recorder)
  ood_score = score_ftn(ood_recorder)
  score = torch.concat([id_score, ood_score])
  is_id = torch.concat([torch.ones_like(id_score), torch.zeros_like(ood_score)])

  fpr, tpr, _ = _binary_roc_compute((score, is_id), None)
  auroc = torch.trapz(tpr, fpr).item()
  plt.scatter(fpr, tpr, label=label_prefix+f' ({auroc:.4f})', **kwargs)
  return auroc


def hist_and_df(runner, score_ftn, title='', bins=100, range=None, alpha=.5, ignores=('FakeData',)):
  id_name = runner.id_name
  ood_names = runner.ood_names
  plt.title(title)
  hist(runner, ood_names[0], True, score_ftn=score_ftn, range=range, bins=bins, alpha=alpha, id_name=id_name, ood_name=ood_names[0])
  for ood in ood_names[1:]:
    if ood in ignores:
      continue
    hist(runner, ood, False, score_ftn=score_ftn, range=range, bins=bins, alpha=alpha, ood_name=ood)
  plt.legend()
  plt.show()
  return metrics_df(runner, score_ftn)