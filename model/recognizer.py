import torch
from torch import nn
from .tanet_adapt import TANetAdapt, TSNHead, TSMHead
from .tsm_adapt import TSMAdapt


class Recognizer(nn.Module):
    def __init__(self, backbone, cls_head):
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head
        self._output_dim = cls_head.fc_cls.in_features

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        x = self.backbone(x)
        x = torch.flatten(x, 1)

        feat, logits = self.cls_head(x)

        if return_feats:
            return feat, logits
        return logits

    @property
    def num_classes(self):
        return self.cls_head.fc_cls.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim


def build_recognizer(model, head, dataset, checkpoint, **kwargs):
    if model == 'tanet':
        if 'ucf101' in dataset:
            num_segments = kwargs.get('num_segments', 16)
            backbone = TANetAdapt(depth=50)
            backbone.init_weights()
            if head == 'tsn':
                cls_head = TSNHead(num_classes=101, in_channels=2048, num_segs=num_segments)
            elif head == 'tsm':
                cls_head = TSMHead(num_classes=101, in_channels=2048, num_segs=num_segments)
            else:
                NotImplementedError
        elif dataset == 'ssv2':
            num_segments = kwargs.get('num_segments', 16)
            backbone = TANetAdapt(depth=50)
            backbone.init_weights()
            if head == 'tsn':
                cls_head = TSNHead(num_classes=174, in_channels=2048, num_segs=num_segments, input_size=256)
            elif head == 'tsm':
                cls_head = TSMHead(num_classes=174, in_channels=2048, num_segs=num_segments, input_size=256)
            else:
                NotImplementedError
        elif dataset == 'mini-ssv2':
            num_segments = kwargs.get('num_segments', 16)
            backbone = TANetAdapt(depth=50)
            backbone.init_weights()
            if head == 'tsn':
                cls_head = TSNHead(num_classes=87, in_channels=2048, num_segs=num_segments, input_size=256)
            elif head == 'tsm':
                cls_head = TSMHead(num_classes=87, in_channels=2048, num_segs=num_segments, input_size=256)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    elif model == 'tsm':
        if 'ucf101' in dataset:
            num_segments = kwargs.get('num_segments', 16)
            backbone = TSMAdapt(num_class=101, num_segments=num_segments, img_feature_dim=224)
            cls_head = TSMHead(num_classes=101, in_channels=2048, num_segs=num_segments, input_size=224)
            state_dict = torch.load(checkpoint)['state_dict']
            model_dict = dict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                model_dict[k] = v
            backbone.load_state_dict(model_dict)
            cls_head.fc_cls.load_state_dict(backbone.new_fc.state_dict())
            backbone = nn.Sequential(*list(backbone.base_model.children())[:-2])
            recognizer = Recognizer(backbone, cls_head)
            return recognizer
        elif dataset == 'ssv2':
            num_segments = kwargs.get('num_segments', 16)
            backbone = TSMAdapt(num_class=174, num_segments=num_segments, img_feature_dim=256)
            cls_head = TSMHead(num_classes=174, in_channels=2048, num_segs=num_segments, input_size=256)
            state_dict = torch.load(checkpoint)['state_dict']
            model_dict = dict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                model_dict[k] = v
            backbone.load_state_dict(model_dict)
            cls_head.fc_cls.load_state_dict(backbone.new_fc.state_dict())
            backbone = nn.Sequential(*list(backbone.base_model.children())[:-2])
            recognizer = Recognizer(backbone, cls_head)
            return recognizer
        elif dataset == 'mini-ssv2':
            ssv2_classes = kwargs.get('ssv2_classes')
            mini_classes = kwargs.get('mini_classes')
            idx = [int(ssv2_classes[k.replace('[', '').replace(']', '')]) for k in mini_classes]
            num_segments = kwargs.get('num_segments', 16)
            backbone = TSMAdapt(num_class=87, num_segments=num_segments, img_feature_dim=256)
            cls_head = TSMHead(num_classes=87, in_channels=2048, num_segs=num_segments, input_size=256)
            state_dict = torch.load(checkpoint)['state_dict']
            model_dict = dict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                if k == 'new_fc.weight' or k == 'new_fc.bias':
                    model_dict[k] = v[idx]
                else:
                    model_dict[k] = v
            backbone.load_state_dict(model_dict)
            cls_head.fc_cls.load_state_dict(backbone.new_fc.state_dict())
            backbone = nn.Sequential(*list(backbone.base_model.children())[:-2])
            recognizer = Recognizer(backbone, cls_head)
            return recognizer
        else:
            raise NotImplementedError
    else:
        raise Exception('Other models are not yet supported!')

    recognizer = Recognizer(backbone, cls_head)

    if 'ucf101' in dataset:
        state_dict = torch.load(checkpoint)
        model_dict = {}
        for k, v in state_dict["state_dict"].items():
            new_k = k.replace('module.base_model', 'backbone') if 'module.base_model' in k else k
            if 'backbone.layer' in new_k and '.net.bn' in new_k:
                id = new_k.index('.net.bn')
                new_k = new_k[0:id] + '.block.conv' + new_k[id + 7] + '.bn' + new_k[id + 8:]
            if 'backbone.layer' in new_k and '.net.conv' in new_k:
                id = new_k.index('.net.conv')
                new_k = new_k[0:id] + '.block.conv' + new_k[id + 9] + '.conv' + new_k[id + 10:]
            if new_k.startswith('backbone.conv'):
                new_k = 'backbone.conv' + new_k[13] + '.conv' + new_k[14:]
            if new_k.startswith('backbone.bn'):
                new_k = 'backbone.conv' + new_k[11] + '.bn' + new_k[12:]
            if 'backbone.layer' in new_k and '.net.downsample.0.' in new_k:
                id = new_k.index('.net.downsample.0.')
                new_k = new_k[0:id] + '.block.downsample.conv' + new_k[id + 17:]
            if 'backbone.layer' in new_k and '.net.downsample.1.' in new_k:
                id = new_k.index('.net.downsample.1.')
                new_k = new_k[0:id] + '.block.downsample.bn' + new_k[id + 17:]
            if new_k == 'module.new_fc.weight':
                new_k = 'cls_head.fc_cls.weight'
            if new_k == 'module.new_fc.bias':
                new_k = 'cls_head.fc_cls.bias'
            model_dict[new_k] = v
    elif dataset == 'ssv2':
        state_dict = torch.load(checkpoint)['state_dict']
        model_dict = dict()
        for k, v in state_dict.items():
            new_k = k.replace('base_model', 'backbone')
            if 'backbone.layer' in new_k and '.net.bn' in new_k:
                id = new_k.index('.net.bn')
                new_k = new_k[0:id] + '.block.conv' + new_k[id + 7] + '.bn' + new_k[id + 8:]
            if 'backbone.layer' in new_k and '.net.conv' in new_k:
                id = new_k.index('.net.conv')
                new_k = new_k[0:id] + '.block.conv' + new_k[id + 9] + '.conv' + new_k[id + 10:]
            if new_k.startswith('backbone.conv'):
                new_k = 'backbone.conv' + new_k[13] + '.conv' + new_k[14:]
            if new_k.startswith('backbone.bn'):
                new_k = 'backbone.conv' + new_k[11] + '.bn' + new_k[12:]
            if 'backbone.layer' in new_k and '.net.downsample.0.' in new_k:
                id = new_k.index('.net.downsample.0.')
                new_k = new_k[0:id] + '.block.downsample.conv' + new_k[id + 17:]
            if 'backbone.layer' in new_k and '.net.downsample.1.' in new_k:
                id = new_k.index('.net.downsample.1.')
                new_k = new_k[0:id] + '.block.downsample.bn' + new_k[id + 17:]
            if new_k == 'new_fc.weight':
                new_k = 'cls_head.fc_cls.weight'
            if new_k == 'new_fc.bias':
                new_k = 'cls_head.fc_cls.bias'
            model_dict[new_k] = v
    elif dataset == 'mini-ssv2':
        ssv2_classes = kwargs.get('ssv2_classes')
        mini_classes = kwargs.get('mini_classes')
        idx = [int(ssv2_classes[k.replace('[', '').replace(']', '')]) for k in mini_classes]
        state_dict = torch.load(checkpoint)['state_dict']
        model_dict = dict()
        for k, v in state_dict.items():
            new_k = k.replace('base_model', 'backbone')
            if 'backbone.layer' in new_k and '.net.bn' in new_k:
                id = new_k.index('.net.bn')
                new_k = new_k[0:id] + '.block.conv' + new_k[id + 7] + '.bn' + new_k[id + 8:]
            if 'backbone.layer' in new_k and '.net.conv' in new_k:
                id = new_k.index('.net.conv')
                new_k = new_k[0:id] + '.block.conv' + new_k[id + 9] + '.conv' + new_k[id + 10:]
            if new_k.startswith('backbone.conv'):
                new_k = 'backbone.conv' + new_k[13] + '.conv' + new_k[14:]
            if new_k.startswith('backbone.bn'):
                new_k = 'backbone.conv' + new_k[11] + '.bn' + new_k[12:]
            if 'backbone.layer' in new_k and '.net.downsample.0.' in new_k:
                id = new_k.index('.net.downsample.0.')
                new_k = new_k[0:id] + '.block.downsample.conv' + new_k[id + 17:]
            if 'backbone.layer' in new_k and '.net.downsample.1.' in new_k:
                id = new_k.index('.net.downsample.1.')
                new_k = new_k[0:id] + '.block.downsample.bn' + new_k[id + 17:]
            if new_k == 'new_fc.weight':
                new_k = 'cls_head.fc_cls.weight'
                model_dict[new_k] = v[idx]
            elif new_k == 'new_fc.bias':
                new_k = 'cls_head.fc_cls.bias'
                model_dict[new_k] = v[idx]
            else:
                model_dict[new_k] = v
    else:
        raise NotImplementedError
    recognizer.load_state_dict(model_dict)
    return recognizer
    