import logging
import click
import gin
from dext.randomization_test.randomization_test import randomization_test
from dext.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", default="config/explainer.gin")
@click.option("--model_name", "-m", help="Model name to explain.",
              default="EFFICIENTDETD0")
@click.option("--explain_mode", default="coco",
              type=click.Choice(["single_image", "coco"]))
@click.option("--dataset_name", default="COCO",
              type=click.Choice(["COCO", "VOC", "MarineDebris"]))
@click.option("--data_split", default="test",  # VOC - train, COCO - test
              type=click.Choice(["test", "train", "val"]))
@click.option("--data_split_name", default="test2017",
              type=click.Choice(["test2017", "train2017", "val2017",
                                 "VOC2012"]))  # VOC - VOC2012, COCO -test2017
@click.option("--input_image_path", default="images/000000309391.jpg")
@click.option("--image_size", default=512)
@click.option("--class_layer_name", default='boxes')
@click.option("--reg_layer_name", default='boxes')
@click.option("--to_explain", default="Classification and Box offset",
              type=click.Choice(["Classification and Box offset",
                                 "Classification", "Box offset"]))
@click.option("--interpretation_method", "-i", default="IntegratedGradients",
              type=click.Choice(["IntegratedGradients", "LRP", "GradCAM",
                                 "GuidedBackpropagation",
                                 "SmoothGrad_IntegratedGradients",
                                 "SmoothGrad_GuidedBackpropagation"]))
@click.option("--visualize_object_index", default=1)  # 1 <
@click.option("--visualize_box_offset", default='x_max',
              type=click.Choice(["y_min", "x_min", "y_max", "x_max"]))
@click.option("--cascade_study", default=False)  # 1 <
@click.option("--randomize_weights_percent", default=0.5)  # 0<=
@click.option("--random_linspace", default=5)  # 1
@click.option("--num_images", default=1)  # 1 <
@click.option("--save_saliency_images", default=True)
@click.option("--save_explanation_images", default=True)
@click.option("--continuous_run", default=False)
@click.option("--explain_top5_backgrounds", default=False)
@click.option("--load_type", default='rgb')
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO",
                                 "DEBUG"]))
@click.option("--log-dir", default="")
def randomizer(config, model_name, explain_mode, dataset_name, data_split,
               data_split_name, input_image_path, image_size, class_layer_name,
               reg_layer_name, to_explain, interpretation_method,
               visualize_object_index, visualize_box_offset, cascade_study,
               randomize_weights_percent, random_linspace, num_images,
               save_saliency_images, save_explanation_images, continuous_run,
               explain_top5_backgrounds, load_type, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    gin.parse_config_file(config)
    LOGGER.info("Running explainer")
    randomization_test(
        model_name, explain_mode, dataset_name, data_split, data_split_name,
        input_image_path, image_size, class_layer_name, reg_layer_name,
        to_explain, interpretation_method, visualize_object_index,
        visualize_box_offset, cascade_study, randomize_weights_percent,
        random_linspace, num_images, save_saliency_images,
        save_explanation_images, continuous_run, explain_top5_backgrounds,
        load_type)


if __name__ == "__main__":
    randomizer()
