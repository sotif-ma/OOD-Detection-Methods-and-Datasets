from typing import Optional
from torch import Tensor
from torchmetrics import CatMetric

class Recorder:
  def __init__(self, features:(tuple['str',...]|dict[str,CatMetric]), compute_on_cpu:bool=True) -> None:
    self.compute_on_cpu=compute_on_cpu
    if isinstance(features, tuple):
      self._recorders = {}
      for k in features:
        self.add_recorder(k)
    else:
      self._recorders = features
      for k, v in features.items():
        setattr(self, k, v)

  def add_recorder(self, name) -> None:
    r = CatMetric(compute_on_cpu=self.compute_on_cpu)
    self._recorders[name] = r
    setattr(self, name, r)

  def update(self, features:dict[str, Tensor]) -> None:
    for k, v in features.items():
      if k in self._recorders.keys():
        self._recorders[k].update(v)
  
  def compute(self) -> dict[str, Tensor]:
    return {k: v.compute() for k, v in self._recorders.items()}

  def __call__(self, features:dict[str, Tensor]) -> dict[str, Tensor]:
    self.update(features)
    return self.compute()
  
  def reset(self):
    for v in self._recorders.values():
      v.reset()

  def clone(self, feature_names:Optional[tuple[str,...]]=None) -> 'Recorder':
    if feature_names is None:
      feature_names = tuple(self._recorders.keys())
    return Recorder({
      nm: self._recorders[nm].clone() for nm in feature_names
    }) # type: ignore
  
  def state_dict(self):
    return {k: v.value for k, v in self._recorders.items()}
  
  def load_state_dict(self, state_dict):
    for k, v in state_dict.items():
      if k in self._recorders.keys():
        self._recorders[k].value = v
        self._recorders[k]._update_count = len(v)