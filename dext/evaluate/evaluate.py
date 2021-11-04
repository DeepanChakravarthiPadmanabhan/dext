import logging
import click
import gin
from dext.evaluate.evaluate_model import evaluate_model
from dext.evaluate.evaluate_explainer import evaluate_explainer
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
@click.option("--result_path", default="evaluation_result.json")
@click.option("--result_dir", default="images/results/")
@click.option("--ap_curve_linspace", default=3)
@click.option("--eval_flip", default=True)
@click.option("--eval_ap_explain", default=False)
@click.option("--merge_saliency_maps", default=True)
@click.option("--merge_method", default='pca',
              type=click.Choice(["pca", "tsne", "and_add", "and_average",
                                 "or_add", "or_average"]))
@click.option("--log_level", default="INFO",
              type=click.Choice(["CRITICAL", "ERROR",
                                 "WARNING", "INFO", "DEBUG"]))
@click.option("--log-dir", default="")
def evaluator(config, mode, model_name, interpretation_method, image_size,
              result_path, result_dir,
              ap_curve_linspace, eval_flip, eval_ap_explain,
              merge_saliency_maps, merge_method, log_level, log_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    gin.parse_config_file(config)
    LOGGER.info("Running explainer")
    if mode == "explainer":
        evaluate_explainer(
            model_name, interpretation_method, image_size, result_path,
            result_dir, ap_curve_linspace,
            eval_flip, eval_ap_explain, merge_saliency_maps, merge_method)
    else:
        evaluate_model(model_name, image_size, result_path)


if __name__ == "__main__":
    evaluator()
