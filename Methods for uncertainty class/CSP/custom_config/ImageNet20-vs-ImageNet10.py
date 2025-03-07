method_name = 'ScalableClassifier'
model_name = 'ViT-B/16'

train_dataset = 'Balance'
custom_name = "Official"
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name).replace('/','_')
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset).replace('/','_')
quick_test = False
# quick_test = True

model = dict(
    type=method_name,
    debug_mode=False,
    t=1,
    ngroup=100,
    group_fuse_num=None,
    classifier=dict(
        type='CLIPScalableClassifier',
        arch=model_name,
        train_dataset='imagenet20',
        # wordnet_database='/data_SSD1/cmy/csp-submit/txtfiles/',
        wordnet_database='/data_SSD1/cmy/NegLabel-official/txtfiles_complete/',
        txt_exclude=
        'noun.person.txt,noun.quantity.txt,noun.group.txt,'
        'adj.pert.txt',
        neg_subsample=-1,
        pos_neg_sim='neg_centroid',
        neg_topk=0.15,  # percentage
        emb_batchsize=1000,
        prompt_idx_pos=85,  # [0,80]
        prompt_idx_neg=85,  # [0,80]
        dump_neg=False,
        load_dump_neg=False,
        pencentile=0.95
    )
)
pipline =[
          dict(type='Collect', keys=['img', 'type'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet20',
        type='MultiFolderDataset',
        path='/data_SSD2/cmy/ImageNet/ILSVRC/Data/CLS-LOC/val_per_class',
        data_ann='./data/ImageNet20.txt',
        pipeline=pipline,
        len_limit=5000 if quick_test else -1
    ),
    ood_data=[
        dict(
            name='ImageNet10',
            type='MultiFolderDataset',
            path='/data_SSD2/cmy/ImageNet/ILSVRC/Data/CLS-LOC/val_per_class',
            data_ann='./data/ImageNet10.txt',
            pipeline=pipline,
            len_limit=5000 if quick_test else -1
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
work_dir = './results/'
