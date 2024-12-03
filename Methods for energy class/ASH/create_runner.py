import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageNet
from torchvision.models import *
from torchvision.models.feature_extraction import create_feature_extractor

import timm

from ood.runner import OODRunner
from ood.dataset import DatasetFilelist, FakeData

observe_id = 'ImageNet'
observe_ood = ['iNaturalist', 'SUN', 'Places', 'Texture', 'OpenImages-O', 'ImageNet-O']

custom_models_and_weights = {
  ### key: (model constructor, transforms, state_dict path)
  "resnet50.A": (resnet50, ResNet50_Weights.IMAGENET1K_V1.transforms(), "./train/resnet50.A.pt"),
  "resnet50.B": (resnet50, ResNet50_Weights.IMAGENET1K_V2.transforms(), "./train/resnet50.B.pt"),
  "resnet50.C": (resnet50, ResNet50_Weights.IMAGENET1K_V1.transforms(), "./train/resnet50.C.pt"),
}

torchvision_models_and_weights = {
  "ResNet50_Weights.IMAGENET1K_V1": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
  "ResNet50_Weights.IMAGENET1K_V2": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
  "ResNet152_Weights.IMAGENET1K_V1": (resnet152, ResNet152_Weights.IMAGENET1K_V1),
  "ResNet152_Weights.IMAGENET1K_V2": (resnet152, ResNet152_Weights.IMAGENET1K_V2),
  "ViT_B_16_Weights.IMAGENET1K_V1": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
  "ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1),
  "ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1),
  "Swin_V2_B_Weights.IMAGENET1K_V1": (swin_v2_b, Swin_V2_B_Weights.IMAGENET1K_V1),
  "MobileNet_V2_Weights.IMAGENET1K_V1": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1),
  "MobileNet_V2_Weights.IMAGENET1K_V2": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V2),
  "MobileNet_V3_Large_Weights.IMAGENET1K_V1": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V1),
  "MobileNet_V3_Large_Weights.IMAGENET1K_V2": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
  "AlexNet_Weights.IMAGENET1K_V1": (alexnet, AlexNet_Weights.IMAGENET1K_V1),
  "ConvNeXt_Base_Weights.IMAGENET1K_V1": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1),
  "DenseNet161_Weights.IMAGENET1K_V1": (densenet161, DenseNet161_Weights.IMAGENET1K_V1),
  "EfficientNet_B7_Weights.IMAGENET1K_V1": (efficientnet_b7, EfficientNet_B7_Weights.IMAGENET1K_V1),
  "EfficientNet_V2_M_Weights.IMAGENET1K_V1": (efficientnet_v2_m, EfficientNet_V2_M_Weights.IMAGENET1K_V1),
  "GoogLeNet_Weights.IMAGENET1K_V1": (googlenet, GoogLeNet_Weights.IMAGENET1K_V1),
  "Inception_V3_Weights.IMAGENET1K_V1": (inception_v3, Inception_V3_Weights.IMAGENET1K_V1),
  "MNASNet1_3_Weights.IMAGENET1K_V1": (mnasnet1_3, MNASNet1_3_Weights.IMAGENET1K_V1),
  "MaxVit_T_Weights.IMAGENET1K_V1": (maxvit_t, MaxVit_T_Weights.IMAGENET1K_V1),
  "RegNet_Y_16GF_Weights.IMAGENET1K_V1": (regnet_y_16gf, RegNet_Y_16GF_Weights.IMAGENET1K_V1),
  "RegNet_Y_16GF_Weights.IMAGENET1K_V2": (regnet_y_16gf, RegNet_Y_16GF_Weights.IMAGENET1K_V2),
  "RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1": (regnet_y_16gf, RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1),
  "RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1": (regnet_y_16gf, RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1),
  "ResNeXt101_32X8D_Weights.IMAGENET1K_V1": (resnext101_32x8d, ResNeXt101_32X8D_Weights.IMAGENET1K_V1),
  "ResNeXt101_32X8D_Weights.IMAGENET1K_V2": (resnext101_32x8d, ResNeXt101_32X8D_Weights.IMAGENET1K_V2),
  "ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1": (shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1),
  "VGG19_BN_Weights.IMAGENET1K_V1": (vgg19_bn, VGG19_BN_Weights.IMAGENET1K_V1),
  "Wide_ResNet101_2_Weights.IMAGENET1K_V1": (wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V1),
  "Wide_ResNet101_2_Weights.IMAGENET1K_V2": (wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V2),
}

timm_models_and_weights = ['eva02_large_patch14_448.mim_m38m_ft_in1k', 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k']

nodes_to_track = {
  ### TorchVision models
  'resnet50': {'avgpool': 'feature', 'fc': 'logit'},
  'resnet152': {'avgpool': 'feature', 'fc': 'logit'},
  'vit_b_16': {'getitem_5': 'feature', 'heads.head': 'logit'},
  'swin_v2_b': {'avgpool': 'feature', 'head': 'logit'},
  'mobilenet_v2': {'adaptive_avg_pool2d': 'feature', 'classifier.1': 'logit'},
  'mobilenet_v3_large': {'classifier.1': 'feature', 'classifier.3': 'logit'},
  'alexnet': {'classifier.5': 'feature', 'classifier.6': 'logit'},
  'convnext_base': {'classifier.1': 'feature', 'classifier.2': 'logit'},
  'densenet161': {'adaptive_avg_pool2d': 'feature', 'classifier': 'logit'},
  'efficientnet_b7': {'avgpool': 'feature', 'classifier.1': 'logit'},
  'efficientnet_v2_m': {'avgpool': 'feature', 'classifier.1': 'logit'},
  'googlenet': {'avgpool': 'feature', 'fc': 'logit'},
  'inception_v3': {'avgpool': 'feature', 'fc': 'logit'},
  'mnasnet1_3': {'mean': 'feature', 'classifier.1': 'logit'},
  'maxvit_t': {'classifier.4': 'feature', 'classifier.5': 'logit'},
  'regnet_y_16gf': {'avgpool': 'feature', 'fc': 'logit'},
  'resnext101_32x8d': {'avgpool': 'feature', 'fc': 'logit'},
  'resnext101_32x8d': {'avgpool': 'feature', 'fc': 'logit'},
  'shufflenet_v2_x2_0': {'mean': 'feature', 'fc': 'logit'},
  'vgg19_bn': {'classifier.4': 'feature', 'classifier.6': 'logit'},
  'wide_resnet101_2': {'avgpool': 'feature', 'fc': 'logit'},
  ### TIMM models
  'eva02_large_patch14_448': {'fc_norm': 'feature', 'head': 'logit'},
}

def get_fancy_name(name:str):
  if name in torchvision_models_and_weights.keys():
    constructor, _ = torchvision_models_and_weights.get(name)
    _, weight_name = name.split('.')
    model_name = constructor.__name__
    weight_name ='_'.join(weight_name.split('_')[1:])
  elif name in timm_models_and_weights:
    model_name, weight_name = name.split('.')
    model_name = model_name.split('_')[0]
    weight_name = weight_name.split('_ft_')[-1]
  elif name in custom_models_and_weights.keys():
    model_name, weight_name = name.split('.')
  else:
    raise NameError(name)
  return model_name + '.' + weight_name

def get_extractor_and_transforms(name:str):
  if name in torchvision_models_and_weights.keys():
    constructor, weights = torchvision_models_and_weights.get(name)
    model = constructor(weights=weights)
    model.eval()
    nodes = nodes_to_track.get(constructor.__name__)
    transforms = weights.transforms()
  elif name in custom_models_and_weights.keys():
    constructor, transforms, fname = custom_models_and_weights.get(name)
    model = constructor()
    model.load_state_dict(torch.load(fname))
    model.eval()
    nodes = nodes_to_track.get(constructor.__name__)
  elif name in timm_models_and_weights:
    model = timm.create_model(name, pretrained=True)
    model.eval()
    nodes = nodes_to_track.get(name.split('.')[0])
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
  else:
    raise NameError(name)
  extractor = create_feature_extractor(model, nodes)
  return extractor, transforms

def prepare_dataloaders(transform, args):
  if hasattr(transform, 'crop_size'):
    ### for TorchVision
    crop_size = transform.crop_size
    if len(crop_size)==1:
      crop_size = (crop_size[0], crop_size[0])
  else:
    ### for TIMM
    crop_size = transform.transforms[1].size

  testdata = {
    'ImageNet': ImageNet(args.imagenet_path, split='val', transform=transform),
    'OpenImages-O': DatasetFilelist(args.openimages_o_path, args.openimages_o_filelist, transform=transform),
    'Texture': DatasetFilelist(args.dtd_path, args.dtd_filelist, transform=transform),
    'iNaturalist': DatasetFilelist(args.inaturalist_path, args.inaturalist_filelist, transform=transform),
    'ImageNet-O': DatasetFilelist(args.imagenet_o_path, args.imagenet_o_filelist, transform=transform),
    'Places': DatasetFilelist(args.places_path, args.places_filelist, transform=transform),
    'SUN': DatasetFilelist(args.sun_path, args.sun_filelist, transform=transform),
    # 'FakeData': FakeData(50000, image_size=(3, *crop_size)),
  }
  testloaders = {
    k: DataLoader(v, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    for k, v in testdata.items()
  }

  id_loader = testloaders[observe_id]
  ood_loaders = tuple(testloaders[k] for k in observe_ood)

  return id_loader, ood_loaders


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Create OOD Runner", add_help=True)
  
  parser.add_argument("weights", type=str, choices=list(torchvision_models_and_weights.keys())+list(custom_models_and_weights.keys())+timm_models_and_weights, help="Model.Weights")
  parser.add_argument("-o", "--output", default=None, type=str, help="path to save outputs")
  parser.add_argument("--device", default="cuda:0", type=str, help="device (Use cuda or cpu Default: cuda:0)")
  parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
  parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
  
  ### Datasets
  parser.add_argument("--imagenet-path", default="./data/imagenet", type=str, help="ImageNet path")
  parser.add_argument("--imagenet-o-path", default="./data/imagenet-o", type=str, help="ImageNet-O path")
  parser.add_argument("--imagenet-o-filelist", default=None, type=str, help="ImageNet-O file list path")
  parser.add_argument("--openimages-o-path", default="./data/open-images/test", type=str, help="OpenImages-O path")
  parser.add_argument("--openimages-o-filelist", default="./data/open-images/openimage_o.txt", type=str, help="OpenImages-O file list path")
  parser.add_argument("--dtd-path", default="./data/dtd/images", type=str, help="Texture path")
  parser.add_argument("--dtd-filelist", default=None, type=str, help="Texture file list path")
  parser.add_argument("--inaturalist-path", default="./data/iNaturalist/images", type=str, help="iNaturalist path")
  parser.add_argument("--inaturalist-filelist", default=None, type=str, help="iNaturalist file list path")
  parser.add_argument("--places-path", default="./data/Places/images", type=str, help="Places path")
  parser.add_argument("--places-filelist", default=None, type=str, help="Places file list path")
  parser.add_argument("--sun-path", default="./data/SUN/images", type=str, help="SUN path")
  parser.add_argument("--sun-filelist", default=None, type=str, help="SUN file list path")

  args = parser.parse_args()
  extractor, transform = get_extractor_and_transforms(args.weights)
  extractor = extractor.to(args.device)
  id_loader, ood_loaders = prepare_dataloaders(transform, args)

  runner = OODRunner()
  runner.run(extractor, id_loader, ood_loaders,
             observe_id, tuple(observe_ood),
             ('feature',), args.device)
  fname = args.output
  if fname is None:
    fname = f'./runners/{get_fancy_name(args.weights)}.pt'
  torch.save(runner.state_dict(), fname)