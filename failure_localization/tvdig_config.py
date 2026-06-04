"""
TVDiag configuration for the SoC-RCA integration.

Matches the GAIA dataset configuration from the original TVDiag project.
"""


class TVDiagConfig:
    """TVDiag model and training configuration."""

    def __init__(
        self,
        # Model dimensions
        alert_embedding_dim: int = 128,
        graph_hidden_dim: int = 64,
        graph_out: int = 32,
        graph_layers: int = 2,
        linear_hidden: list = None,
        feat_drop: float = 0.0,
        aggregator: str = "mean",
        # Training
        epochs: int = 500,
        batch_size: int = 512,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        patience: int = 10,
        temperature: float = 0.3,
        contrastive_loss_scale: float = 0.1,
        # TVDiag modules
        TO: bool = True,
        CM: bool = True,
        dynamic_weight: bool = True,
        # Augmentation
        aug_percent: float = 0.2,
        aug_times: int = 10,
        # GAIA specifics
        ft_num: int = 5,
        modalities: list = None,
        # Paths
        model_checkpoint_dir: str = "",
        data_dir: str = "",
    ):
        self.alert_embedding_dim = alert_embedding_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_out = graph_out
        self.graph_layers = graph_layers
        self.linear_hidden = linear_hidden or [64]
        self.feat_drop = feat_drop
        self.aggregator = aggregator
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.temperature = temperature
        self.contrastive_loss_scale = contrastive_loss_scale
        self.TO = TO
        self.CM = CM
        self.dynamic_weight = dynamic_weight
        self.aug_percent = aug_percent
        self.aug_times = aug_times
        self.ft_num = ft_num
        self.modalities = modalities or ["metric", "trace", "log"]
        self.model_checkpoint_dir = model_checkpoint_dir
        self.data_dir = data_dir

    # ----- GAIA static topology -----

    NODE_NAMES = [
        "dbservice1", "dbservice2",
        "logservice1", "logservice2",
        "mobservice1", "mobservice2",
        "redisservice1", "redisservice2",
        "webservice1", "webservice2",
    ]

    # Edges extracted from TVDiag data/gaia/raw/edges.json (typical graph)
    # Format: list of [src_idx, dst_idx]
    GAIA_EDGES = [
        [2, 0], [3, 0], [3, 1], [2, 1],  # logservice -> dbservice
        [8, 2], [3, 2], [9, 2], [8, 3],   # webservice -> logservice
        [9, 3], [2, 3], [8, 4], [9, 4],   # webservice -> logservice/mobservice
        [8, 5], [9, 5],                     # webservice -> mobservice
        [4, 6], [8, 6], [5, 6], [3, 6],   # mobservice/logservice -> redisservice
        [0, 6], [1, 6], [9, 6], [2, 6],   # dbservice/logservice -> redisservice
        [3, 7], [4, 7], [8, 7], [2, 7],   # logservice/mobservice -> redisservice
        [1, 7], [9, 7], [5, 7], [0, 7],   # dbservice/mobservice -> redisservice
        [5, 0], [0, 5],                     # mobservice <-> dbservice
        [1, 3],                             # dbservice2 -> logservice2
        [7, 1], [7, 3],                     # redisservice2 -> dbservice/logservice
        [2, 9],                             # logservice1 -> webservice2
        [6, 4],                             # redisservice1 -> mobservice1
        [4, 8],                             # mobservice1 -> webservice1
        [6, 8],                             # redisservice1 -> webservice1
    ]

    def print_configs(self, logger):
        for attr, value in vars(self).items():
            logger.info(f"  {attr}: {value}")
