import logging
import click
import gin
from dext.error_analyzer.error_analysis import analyze_errors
from dext.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", default="config/explainer.gin")
@click.option("--model_name", "-m", help="Model name to explain.",
              default="EFFICIENTDETD0")
@click.option("--explain_mode", default="dataset",
              type=click.Choice(["single_image", "dataset"]))
@click.option("--dataset_name", default="VOC",
              type=click.Choice(["COCO", "VOC"]))
@click.option("--data_split", default="train",  # VOC - train, COCO - test
              type=click.Choice(["test", "train", "val"]))
@click.option("--data_split_name", default="VOC2012",
              type=click.Choice(["test2017", "train2017", "val2017",
                                 "VOC2012"]))  # VOC - VOC2012, COCO -test2017
@click.option("--input_image_path", default="images/000000309391.jpg")
@click.option("--image_size", default=512)
@click.option("--class_layer_name", default='boxes')
@click.option("--reg_layer_name", default='boxes')
@click.option("--to_explain", default="Boxoffset",
              type=click.Choice(["Classification", "Boxoffset"]))
@click.option("--interpretation_method", "-i", default="IntegratedGradients",
              type=click.Choice(["IntegratedGradients", "LRP", "GradCAM",
                                 "GuidedBackpropagation",
                                 "SmoothGrad_GuidedBackpropagation",
                                 "SmoothGrad_IntegratedGradients", ]))
@click.option("--visualize_object_index", default=0)  # 1 <
@click.option("--visualize_box_offset", default='x_max',
              type=click.Choice(["y_min", "x_min", "y_max", "x_max"]))
@click.option("--visualize_class", default='dog')
@click.option("--num_images", default=1)  # 1 <
@click.option("--save_saliency_images", default=True)
@click.option("--save_explanation_images", default=True)
@click.option("--continuous_run", default=False)
@click.option("--plot_gt", default=True)
@click.option("--analyze_error_type", default='missed',
              type=click.Choice(['missed', 'wrong_class',
                                 'poor_localization']))
@click.option("--use_own_class", default=False)
@click.option("--saliency_threshold", default=None)
@click.option("--grad_times_input", default=False)
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO",
                                 "DEBUG"]))
@click.option("--log-dir", default="")
def explainer(config, model_name, explain_mode, dataset_name, data_split,
              data_split_name, input_image_path, image_size, class_layer_name,
              reg_layer_name, to_explain, interpretation_method,
              visualize_object_index, visualize_box_offset, visualize_class,
              num_images, save_saliency_images, save_explanation_images,
              continuous_run, plot_gt, analyze_error_type, use_own_class,
              saliency_threshold, grad_times_input, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    gin.parse_config_file(config)
    LOGGER.info("Running explainer")
    analyze_errors(model_name, explain_mode, dataset_name, data_split,
                   data_split_name, input_image_path, image_size,
                   class_layer_name, reg_layer_name, to_explain,
                   interpretation_method, visualize_object_index,
                   visualize_box_offset, visualize_class, num_images,
                   save_saliency_images, save_explanation_images,
                   continuous_run, plot_gt, analyze_error_type, use_own_class,
                   saliency_threshold, grad_times_input)


if __name__ == "__main__":
    explainer()
