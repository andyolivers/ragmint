from ragmint.tuner import RAGMint
from ragmint.utils.logger import Logger


def test_tuner_instantiation():
    logger = Logger()
    tuner = RAGMint(config_path="configs/default.yaml", logger=logger)
    assert tuner is not None
