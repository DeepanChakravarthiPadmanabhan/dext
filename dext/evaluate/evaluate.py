import logging
import click
import gin
from dext.evaluate.evaluate_model import evaluate_model
from dext.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", default="config/evaluator.gin")
@click.option("--model_name", "-m", help="Model name to explain.",
              default="EFFICIENTDETD0")
@click.option("--result_path", default="evaluation_result.json")
@click.option("--image_size", default=512)
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR",
                                 "WARNING", "INFO", "DEBUG"]))
@click.option("--log-dir", default="")
def evaluator(config, model_name, image_size, result_path, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    gin.parse_config_file(config)
    LOGGER.info("Running explainer")
    evaluate_model(model_name, image_size, result_path)


if __name__ == "__main__":
    evaluator()
