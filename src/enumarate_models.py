import logging; LOGGER = logging.getLogger("enumuration")
import torch
import re

import config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    models = [(x,x.name) for x in config.MODEL_FOLDER.iterdir() if x.is_file() and x.name.endswith(".pth")]
    models = sorted(models, key=lambda x: int(re.search(r'\d+', x[1]).group())) # pyright: ignore[reportOptionalMemberAccess]
    if ( len(models) == 0 ):
        LOGGER.critical("Model folder is empty!")
        raise FileNotFoundError

    LOGGER.info("Model fodler contains:")
    for model_path in models:
        checkpoint = torch.load( model_path[0].absolute(), map_location=config.DEVICE, weights_only=True )
        LOGGER.info(f"{model_path[1]}, epoch {checkpoint["epoch"]}, best_validation_loss {checkpoint["best_validation_loss"]}")
    LOGGER.info("End")