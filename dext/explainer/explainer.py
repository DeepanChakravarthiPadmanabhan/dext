import logging
import click
from dext.explainer.explain_model import explain_model
from dext.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--model_name", "-m", help="Model name to explain.",
              default="EFFICIENTDETD0")
@click.option("--input_image_path",
              default="images/000000309391.jpg")
@click.option("--interpretation_method", default="IntegratedGradients")
@click.option("--layer_name", default=None)
@click.option("--visualize_object", default=None)
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR",
                                 "WARNING", "INFO", "DEBUG"]))
@click.option("--log-dir", default="")
def explainer(model_name, input_image_path, interpretation_method, layer_name,
              visualize_object, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    LOGGER.info("Running explainer")
    explain_model(model_name, input_image_path,
                  interpretation_method, layer_name, visualize_object)


if __name__ == "__main__":
    explainer()
