from .resnet_fpn import ResNetFPN

def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        return ResNetFPN(config['resnetfpn'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
