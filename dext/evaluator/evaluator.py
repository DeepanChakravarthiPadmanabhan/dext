import logging
import click
import gin
from dext.evaluator.evaluate_model import evaluate_model
from dext.evaluator.evaluate_explainer import evaluate_explainer
from dext.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", default="config/evaluator.gin")
@click.option("--mode", default="explainer")
@click.option("--model_name", "-m", help="Model name to explain.",
              default="EFFICIENTDETD0")
@click.option("--interpretation_method", "-i", default="IntegratedGradients",
              type=click.Choice(["IntegratedGradients", "SmoothGrad", "LRP",
                                 "GuidedBackpropagation", "GradCAM"]))
@click.option("--image_size", default=512)
@click.option("--results_dir", default="images/results/")
@click.option("--num_images", default=100)  # 1 <
@click.option("--ap_curve_linspace", default=100)
@click.option("--eval_flip", default=True)
@click.option("--eval_ap_explain", default=False)
@click.option("--merge_saliency_maps", default=True)
@click.option("--merge_method", default='pca',
              type=click.Choice(["pca", "tsne", "and_add", "and_average",
                                 "or_add", "or_average"]))
@click.option("--save_modified_images", default=False)
@click.option("--coco_result_file", default="evaluation_result.json")
@click.option("--image_adulteration_method", default='inpainting',
              type=click.Choice(["inpainting", "zeroing"]))
@click.option("--explain_top5_backgrounds", default=False)
@click.option("--continuous_run", default=False)
@click.option("--save_all_map_metrics", default=True)
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR",
                                 "WARNING", "INFO", "DEBUG"]))
@click.option("--log-dir", default="")
def evaluator(config, mode, model_name, interpretation_method, image_size,
              results_dir, num_images, ap_curve_linspace, eval_flip,
              eval_ap_explain, merge_saliency_maps, merge_method,
              save_modified_images, coco_result_file,
              image_adulteration_method, explain_top5_backgrounds,
              continuous_run, save_all_map_metrics, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    gin.parse_config_file(config)
    LOGGER.info("Running explainer")
    if mode == "explainer":
        evaluate_explainer(
            model_name, interpretation_method, image_size, results_dir,
            num_images, ap_curve_linspace, eval_flip, eval_ap_explain,
            merge_saliency_maps, merge_method, save_modified_images,
            coco_result_file, image_adulteration_method,
            explain_top5_backgrounds, continuous_run, save_all_map_metrics)
    else:
        evaluate_model(model_name, image_size, coco_result_file)


if __name__ == "__main__":
    evaluator()
