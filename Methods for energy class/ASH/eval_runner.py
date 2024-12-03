import torch
import timm
import pandas as pd

from ood.runner import OODRunner
from ood.scores import msp, energy, trim
from ood.prune import dice, ash_s
from ood.metrics import metrics_df
from create_runner import torchvision_models_and_weights, timm_models_and_weights, custom_models_and_weights, nodes_to_track, get_fancy_name, observe_ood

observe_metric = ['FPR@95', 'AUROC']


_nodes = {k1: {v2: k2 for k2, v2 in v1.items()} for k1, v1 in nodes_to_track.items()}

def get_last_layer(name:str) -> torch.nn.Linear:
  if name in torchvision_models_and_weights.keys():
    constructor, weights = torchvision_models_and_weights.get(name)
    model = constructor(weights=weights)
    model.eval()
    model_name = constructor.__name__
  elif name in custom_models_and_weights.keys():
    constructor, _, weights = custom_models_and_weights.get(name)
    model = constructor()
    model.load_state_dict(torch.load(weights))
    model.eval()
    model_name = constructor.__name__
  elif name in timm_models_and_weights:
    model = timm.create_model(name, pretrained=True)
    model.eval()
    model_name = name.split('.')[0]
  else:
    raise NameError(name)

  layer_name = _nodes.get(model_name).get('logit').split('.')
  if len(layer_name) == 1:
    return model._modules[layer_name[0]]
  elif len(layer_name) == 2:
    return model._modules[layer_name[0]]._modules[layer_name[1]]
  else:
    raise

def get_df(runner:OODRunner, last_layer:torch.nn.Linear) -> tuple[float, pd.DataFrame]:
  _true = runner.id_recorder.compute()['_class']
  _pred = runner.id_recorder.compute()['_logit'].max(dim=-1)[1]
  id_acc = (_true == _pred).type(torch.float).mean().item()

  msp_df = metrics_df(runner, msp).loc[observe_ood, observe_metric]
  col_idx = pd.MultiIndex.from_tuples([('MSP', 'FPR@95'), ('MSP', 'AUROC')], names=('Method', 'Metric'))
  msp_df.columns = col_idx

  energy_df = metrics_df(runner, energy).loc[observe_ood, observe_metric]
  col_idx = pd.MultiIndex.from_tuples([('Energy', 'FPR@95'), ('Energy', 'AUROC')], names=('Method', 'Metric'))
  energy_df.columns = col_idx

  dice_runner = dice(runner, last_layer, percentile=90)
  dice_df = metrics_df(dice_runner, energy).loc[observe_ood, observe_metric]
  col_idx = pd.MultiIndex.from_tuples([('DICE', 'FPR@95'), ('DICE', 'AUROC')], names=('Method', 'Metric'))
  dice_df.columns = col_idx
  del dice_runner

  ash_runner = ash_s(runner, last_layer, percentile=90)
  ash_df = metrics_df(ash_runner, energy).loc[observe_ood, observe_metric]
  col_idx = pd.MultiIndex.from_tuples([('ASH_S', 'FPR@95'), ('ASH_S', 'AUROC')], names=('Method', 'Metric'))
  ash_df.columns = col_idx
  del ash_runner

  trim_df = metrics_df(runner, trim).loc[observe_ood, observe_metric]
  col_idx = pd.MultiIndex.from_tuples([('TRIM', 'FPR@95'), ('TRIM', 'AUROC')], names=('Method', 'Metric'))
  trim_df.columns = col_idx

  return id_acc, pd.concat([msp_df, energy_df, dice_df, ash_df, trim_df], axis=1)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Evaluate OOD Runner", add_help=True)
  
  parser.add_argument("weights", type=str, choices=list(torchvision_models_and_weights.keys())+list(custom_models_and_weights.keys())+timm_models_and_weights, help="Model.Weights")
  parser.add_argument("-i", "--input", default=None, type=str, help="path to saved runner")
  parser.add_argument("-o", "--output", default=None, type=str, help="path to save the result in CSV format")
  
  args = parser.parse_args()

  ifname, ofname = args.input, args.output
  fname = get_fancy_name(args.weights)
  if ifname is None:
    ifname = './runners/' + fname + '.pt'
  if ofname is None:
    ofname = './results/' + fname + '.csv'
  acc_fname = '.'.join(ofname.split('.')[:-1]) + '.acc'

  runner = OODRunner(ifname)
  last_layer = get_last_layer(args.weights)

  id_acc, df = get_df(runner, last_layer)
  df.to_csv(ofname)
  with open(acc_fname, 'w') as f:
    f.write(str(id_acc))

  print(fname, f"(ID Accuracy: {id_acc:.6f})")
  print(df)