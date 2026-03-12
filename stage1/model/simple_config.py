class SimpleSidaConfig:
    def __init__(
        self,
        train_mask_decoder=True,
        out_dim=256,
        cls_loss_weight=1.0,
        mask_loss_weight=1.0,
        ce_loss_weight=1.0,
        dice_loss_weight=1.0,
        bce_loss_weight=1.0,
        vision_pretrained="PATH_TO_SAM_ViT-H",
        vision_tower="openai/clip-vit-large-patch14",
        use_mm_start_end=True,
        num_classes=3,
        image_size=1024,
    ):
        self.train_mask_decoder = train_mask_decoder
        self.out_dim = out_dim
        self.cls_loss_weight = cls_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.vision_pretrained = vision_pretrained
        self.vision_tower = vision_tower
        self.use_mm_start_end = use_mm_start_end
        self.num_classes = num_classes
        self.image_size = image_size



class SimpleSidaLocalizationConfig:
    def __init__(
        self,
        train_mask_decoder=True,
        out_dim=256,
        mask_loss_weight=1.0,
        ce_loss_weight=1.0,
        dice_loss_weight=1.0,
        bce_loss_weight=1.0,
        vision_pretrained="PATH_TO_SAM_ViT-H",
        vision_tower="openai/clip-vit-large-patch14",
        use_mm_start_end=True,
        image_size=1024,
       
        **kwargs
    ):
        self.train_mask_decoder = train_mask_decoder
        self.out_dim = out_dim
        self.mask_loss_weight = mask_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.vision_pretrained = vision_pretrained
        self.vision_tower = vision_tower
        self.use_mm_start_end = use_mm_start_end
        self.image_size = image_size