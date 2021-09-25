import logging
import click
from dext.evaluate.evaluate_model import evaluate_model
from dext.utils import setup_logging
from dext.utils.constants import COCO_VAL_ANNOTATION_FILE
from dext.utils.constants import DATASET_PATH

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--model_name", "-m", help="Model name to explain.",
              default="EFFICIENTDETD0")
@click.option("--result_path", default="evaluation_result.json")
@click.option("--dataset_path", default=DATASET_PATH)
@click.option("--annotation_path", default=COCO_VAL_ANNOTATION_FILE)
@click.option("--image_size", default=512)
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR",
                                 "WARNING", "INFO", "DEBUG"]))
@click.option("--log-dir", default="")
def evaluator(model_name, image_size, dataset_path, annotation_path,
              result_path, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    LOGGER.info("Running explainer")
    evaluate_model(model_name, image_size, dataset_path, annotation_path,
                   result_path)


if __name__ == "__main__":
    evaluator()
