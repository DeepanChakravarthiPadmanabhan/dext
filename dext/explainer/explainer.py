import logging
import click
import gin
from dext.explainer.explain_model import explain_model
from dext.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", default="config/explainer.gin")
@click.option("--model_name", "-m", help="Model name to explain.",
              default="EFFICIENTDETD0")
@click.option("--explain_mode", default="coco",
              type=click.Choice(["single_image", "coco"]))
@click.option("--input_image_path", default="images/000000309391.jpg")
@click.option("--image_size", default=512)
@click.option("--class_layer_name", default='boxes')
@click.option("--reg_layer_name", default='boxes')
@click.option("--to_explain", default="Classification and Box offset",
              type=click.Choice(["Classification and Box offset",
                                 "Classification", "Box offset"]))
@click.option("--result_dir", default="/scratch/dpadma2s/thesis/results/")
@click.option("--interpretation_method", "-i", default="IntegratedGradients",
              type=click.Choice(["IntegratedGradients", "SmoothGrad", "LRP",
                                 "GuidedBackpropagation", "GradCAM"]))
@click.option("--visualize_object_index", default='all')  # 1 <
@click.option("--visualize_box_offset", default='x_max',
              type=click.Choice(["y_min", "x_min", "y_max", "x_max"]))
@click.option("--num_images", default=5)  # 1 <
@click.option("--save_saliency_images", default=True)
@click.option("--save_explanation_images", default=True)
@click.option("--continuous_run", default=True)
@click.option("--explain_top5_backgrounds", default=False)
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO",
                                 "DEBUG"]))
@click.option("--log-dir", default="")
def explainer(config, model_name, explain_mode, input_image_path, image_size,
              class_layer_name, reg_layer_name, to_explain, result_dir,
              interpretation_method, visualize_object_index,
              visualize_box_offset, num_images, save_saliency_images,
              save_explanation_images, continuous_run,
              explain_top5_backgrounds, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    gin.parse_config_file(config)
    LOGGER.info("Running explainer")
    explain_model(model_name, explain_mode, input_image_path, image_size,
                  class_layer_name, reg_layer_name, to_explain, result_dir,
                  interpretation_method, visualize_object_index,
                  visualize_box_offset, num_images, save_saliency_images,
                  save_explanation_images, continuous_run,
                  explain_top5_backgrounds)


if __name__ == "__main__":
    explainer()
