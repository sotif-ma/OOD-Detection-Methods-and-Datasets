import os
from typing import Any, Callable, Optional, Dict, List, Tuple

import torch
from torchvision.datasets.folder import DatasetFolder, default_loader
from torchvision.datasets.vision import VisionDataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class DatasetFilelist(DatasetFolder):
  def __init__(
    self,
    img_root: str,
    filelist_path: Optional[str] = None,
    loader: Callable[[str], Any] = default_loader,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
  ) -> None:
    super(DatasetFolder, self).__init__(img_root, transform=transform, target_transform=target_transform)
    classes, class_to_idx = self.find_classes(self.root)
    samples = self.make_dataset(self.root, class_to_idx, filelist_path, is_valid_file)

    self.loader = loader

    self.classes = classes
    self.class_to_idx = class_to_idx
    self.samples = samples
    self.targets = [s[1] for s in samples]

  @staticmethod
  def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    filelist: Optional[str] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
  ) -> List[Tuple[str, int]]:
    if is_valid_file is None:
      def is_valid_file(x: str) -> bool:
        return '.'+x.split('.')[-1].lower() in IMG_EXTENSIONS 
    if class_to_idx is None:
      # prevent potential bug since make_dataset() would use the class_to_idx logic of the
      # find_classes() function, instead of using that of the find_classes() method, which
      # is potentially overridden and thus could have a different logic.
        raise ValueError("The class_to_idx parameter cannot be None.")
    res = []
    if filelist is None:
      for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
          if is_valid_file(fname):
            res.append((os.path.join(root, fname), 0))
      return res
    with open(filelist, 'r') as f:
      for l in f.readlines():
        fname_and_label = l.strip().rsplit(maxsplit=1)
        if len(fname_and_label) == 2:
          fname, label = fname_and_label
        else:
          fname, label = fname_and_label[0], 0
        if is_valid_file(fname):
          res.append((os.path.join(directory,fname), int(label)))
    return res

  @staticmethod
  def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        return [''], {'': 0}
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class FakeData(VisionDataset):
    """A fake dataset that returns randomly generated images and returns them as PIL images

    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the dataset. Default: 10
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0

    """

    def __init__(
        self,
        size: int = 1000,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 10,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        random_offset: int = 0,
    ) -> None:
        super().__init__(None, transform=transform, target_transform=target_transform)  # type: ignore[arg-type]
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError(f"{self.__class__.__name__} index out of range")
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        # img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target.item()

    def __len__(self) -> int:
        return self.size
