import logging
import click
import gin
from dext.trust_analyzer.sample_generation import sample_generator
from dext.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", default="config/explainer.gin")
@click.option("--generate_samples", default=True)
@click.option("--model_name", "-m", help="Model name to explain.",
              default="all")
@click.option("--explain_mode", default="dataset",
              type=click.Choice(["single_image", "dataset"]))
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
                                 "Classification", "Boxoffset"]))
@click.option("--visualize_object_index", default='all')  # 1 <
@click.option("--visualize_box_offset", default='x_max',
              type=click.Choice(["y_min", "x_min", "y_max", "x_max"]))
@click.option("--num_images", default=10)  # 1 <
@click.option("--continuous_run", default=False)
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO",
                                 "DEBUG"]))
@click.option("--log-dir", default="")
def trust_analyzer(config, generate_samples, model_name, explain_mode,
                   dataset_name, data_split, data_split_name, input_image_path,
                   image_size, class_layer_name, reg_layer_name, to_explain,
                   visualize_object_index, visualize_box_offset, num_images,
                   continuous_run, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    gin.parse_config_file(config)
    LOGGER.info("Running explainer")
    if generate_samples:
        sample_generator(model_name, explain_mode, dataset_name, data_split,
                         data_split_name, input_image_path, image_size,
                         class_layer_name, reg_layer_name, to_explain,
                         visualize_object_index, visualize_box_offset,
                         num_images, continuous_run)


if __name__ == "__main__":
    trust_analyzer()
