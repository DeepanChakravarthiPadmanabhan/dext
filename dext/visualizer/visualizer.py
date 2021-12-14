import logging
import click
import gin
from dext.visualizer.multi_detection_visualizer import (
    multi_detection_visualizer)
from dext.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", default="config/evaluator.gin")
@click.option("--model_name", "-m", help="Model name to explain.",
              default="EFFICIENTDETD0")
@click.option("--interpretation_method", "-i", default="IntegratedGradients",
              type=click.Choice(["IntegratedGradients", "SmoothGrad", "LRP",
                                 "GuidedBackpropagation", "GradCAM"]))
@click.option("--image_size", default=512)
@click.option("--results_dir", default="images/results/")
@click.option("--num_images", default=1)  # 1 <
@click.option("--continuous_run", default=True)
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR",
                                 "WARNING", "INFO", "DEBUG"]))
@click.option("--log-dir", default="")
def visualizer(config, model_name, interpretation_method, image_size,
               results_dir, num_images, continuous_run, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    gin.parse_config_file(config)
    LOGGER.info("Running explainer")
    multi_detection_visualizer(
        model_name, interpretation_method, image_size, results_dir,
        num_images, continuous_run)

if __name__ == "__main__":
    visualizer()
