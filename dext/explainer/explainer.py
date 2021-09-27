import logging
import click
from dext.explainer.explain_model import explain_model
from dext.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--model_name", "-m", help="Model name to explain.",
              default="EFFICIENTDETD0")
@click.option("--explain_mode", default="coco")
@click.option("--input_image_path", default="images/000000309391.jpg")
@click.option("--image_size", default=512)
@click.option("--class_layer_name", default='boxes')
@click.option("--reg_layer_name", default='boxes')
@click.option("--to_explain", default="Classification and Box offset")
@click.option("--result_dir", default="images/results/")
@click.option("--interpretation_method", default="IntegratedGradients")
@click.option("--visualize_object_index", default='all')  # 1 <
@click.option("--visualize_box_offset", default='y_min')
@click.option("--num_images", default=1)  # 1 <
@click.option("--merge_method", default='add')
@click.option("--save_detections", default=False)
@click.option("--save_explanations", default=True)
@click.option("--analyze_each_maps", default=False)
@click.option("--ap_curve_linspace", default=20)
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR",
                                 "WARNING", "INFO", "DEBUG"]))
@click.option("--log-dir", default="")
def explainer(model_name, explain_mode, input_image_path, image_size,
              class_layer_name, reg_layer_name, to_explain, result_dir,
              interpretation_method, visualize_object_index,
              visualize_box_offset, num_images, merge_method, save_detections,
              save_explanations, analyze_each_maps, ap_curve_linspace,
              log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    LOGGER.info("Running explainer")
    explain_model(model_name, explain_mode, input_image_path, image_size,
                  class_layer_name, reg_layer_name, to_explain, result_dir,
                  interpretation_method, visualize_object_index,
                  visualize_box_offset, num_images, merge_method,
                  save_detections, save_explanations, analyze_each_maps,
                  ap_curve_linspace)


if __name__ == "__main__":
    explainer()
