from typing import Optional
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .recorder import Recorder

def run_over(model:torch.nn.Module,
             dataloader:DataLoader,
             logit_node_name:str='logit',
             recorder:Optional[Recorder]=None,
             prog_bar_desc:Optional[str]=None,
             device:str='cuda:0') -> None:
  """Run model over a dataloader.

  Args:
    model: a model to run which returns logit.
    dataloader: a dataloader.
    logit_node_name: in the case of the `model` is an extractor(`torch.fx.GraphModule`), it indicates logit node's name. If `model` is an module (`torch.nn.module`), then it will be ignored.
    recorder: a `Recorder` instance to recorde features, predictions and targets.
    prog_bar_desc: `tqdm` desc.
    device: device.

  Returns:
    Returns `None`. This function just updates `recorder`.
  """
  for x, y in tqdm(dataloader, desc=prog_bar_desc):
    with torch.no_grad():
      res = model(x.to(device))
      if isinstance(model, torch.fx.GraphModule):
        logit = res[logit_node_name]
        if recorder is not None:
          recorder.update({k:v for k,v in res.items() if k != logit_node_name})
      else:
        logit = res
      if recorder is not None:
        recorder.update({'_class':y, '_logit':logit})

class OODRunner:
  """OOD Runner.

  Attributes:
    model: a model to run which returns logit.
    id_name: the name of ID dataset.
    ood_names: the names of OOD datasets.
    id_recorder: ID recorder
    ood_recorders: `dict` of OOD recorders
    _ran: `True` if it had been ran.
  """
  def __init__(self, file_path:str=None):
    self._ran = False
    self.id_recorder = None
    self.ood_recorders = {}
    if file_path:
      self.load_state_dict(torch.load(file_path))

  def run(self,
          model:torch.nn.Module,
          id_loader:DataLoader,
          ood_loaders:tuple[DataLoader],
          id_name:str='ID',
          ood_names:Optional[tuple[str,...]]=None,
          record:tuple[str,...]=tuple(),
          device='cuda:0') -> None:
    assert model is not None
    self.id_name = id_name
    if ood_names is None:
      ood_names = tuple(f'OOD{i}' for i in range(len(ood_loaders)))
    assert len(ood_names) == len(ood_loaders)
    self.ood_names = ood_names
    record = record + ('_class', '_logit')
    self.id_recorder = Recorder(record, compute_on_cpu=True)
    run_over(model=model,
             dataloader=id_loader,
             recorder=self.id_recorder,
             prog_bar_desc=id_name,
             device=device)
    self.ood_recorders = {}
    for (nm, ood_loader) in zip(ood_names, ood_loaders):
      self.ood_recorders[nm] = Recorder(record, compute_on_cpu=True)
      run_over(model=model,
               dataloader=ood_loader,
               recorder=self.ood_recorders[nm],
               prog_bar_desc=nm,
               device=device)
    self._ran = True

  def state_dict(self):
    assert self._ran == True
    return {
      'id_name': self.id_name,
      'id_recorder': self.id_recorder.state_dict(),
      'ood_names': self.ood_names,
      'ood_recorders': {k: v.state_dict() for k, v in self.ood_recorders.items()}
    }
  
  def load_state_dict(self, state_dict):
    self.id_name = state_dict['id_name']
    self.ood_names = state_dict['ood_names']
    record = tuple(state_dict['id_recorder'].keys())
    self.id_recorder = Recorder(record, compute_on_cpu=True)
    self.id_recorder.load_state_dict(state_dict['id_recorder'])
    for k, v in state_dict['ood_recorders'].items():
      self.ood_recorders[k] = Recorder(record, compute_on_cpu=True)
      self.ood_recorders[k].load_state_dict(v)
    self._ran = True

  def clone(self):
    new_runner = OODRunner()
    new_runner.load_state_dict(self.state_dict())
    return new_runner