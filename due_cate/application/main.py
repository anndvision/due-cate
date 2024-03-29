import ray
import click

from torch import cuda

from pathlib import Path

from due_cate.application import workflows


@click.group(chain=True)
@click.pass_context
def cli(context):
    context.obj = {"n_gpu": cuda.device_count()}


@cli.command("tune")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option(
    "--max-samples",
    default=100,
    type=int,
    help="maximum number of search space samples, default=100",
)
@click.option(
    "--gpu-per-trial",
    default=0.0,
    type=float,
    help="number of gpus for each trial, default=0",
)
@click.option(
    "--cpu-per-trial",
    default=1.0,
    type=float,
    help="number of cpus for each trial, default=1",
)
@click.option(
    "--seed",
    default=1331,
    type=int,
    help="random number generator seed, default=1331",
)
@click.pass_context
def tune(
    context,
    job_dir,
    max_samples,
    gpu_per_trial,
    cpu_per_trial,
    seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    gpu_per_trial = 0 if context.obj["n_gpu"] == 0 else gpu_per_trial
    context.obj.update(
        {
            "job_dir": job_dir,
            "max_samples": max_samples,
            "gpu_per_trial": gpu_per_trial,
            "cpu_per_trial": cpu_per_trial,
            "seed": seed,
            "tune": True,
        }
    )


@cli.command("train")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option("--num-trials", default=1, type=int, help="number of trials, default=1")
@click.option(
    "--gpu-per-trial",
    default=0.0,
    type=float,
    help="number of gpus for each trial, default=0",
)
@click.option(
    "--cpu-per-trial",
    default=1.0,
    type=float,
    help="number of cpus for each trial, default=1",
)
@click.option("--verbose", default=False, type=bool, help="verbosity default=False")
@click.option(
    "--seed",
    default=1331,
    type=int,
    help="random number generator seed, default=1331",
)
@click.pass_context
def train(
    context,
    job_dir,
    num_trials,
    gpu_per_trial,
    cpu_per_trial,
    verbose,
    seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    gpu_per_trial = 0 if context.obj["n_gpu"] == 0 else gpu_per_trial
    context.obj.update(
        {
            "job_dir": job_dir,
            "num_trials": num_trials,
            "gpu_per_trial": gpu_per_trial,
            "cpu_per_trial": cpu_per_trial,
            "verbose": verbose,
            "seed": seed,
            "tune": False,
        }
    )


@cli.command("evaluate")
@click.option(
    "--experiment-dir",
    type=str,
    required=True,
    help="location for reading checkpoints",
)
@click.option(
    "--output-dir",
    type=str,
    required=False,
    default=None,
    help="location for writing results",
)
@click.pass_context
def evaluate(
    context,
    experiment_dir,
    output_dir,
):
    output_dir = experiment_dir if output_dir is None else output_dir
    context.obj.update(
        {
            "experiment_dir": experiment_dir,
            "output_dir": output_dir,
        }
    )
    workflows.evaluation.evaluate(
        experiment_dir=Path(experiment_dir),
        output_dir=Path(output_dir),
    )


@cli.command("ihdp")
@click.pass_context
@click.option(
    "--root",
    type=str,
    required=True,
    help="location of dataset",
)
def ihdp(
    context,
    root,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "ihdp"
    experiment_dir = job_dir / dataset_name
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "root": root,
                "split": "train",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
            "ds_valid": {
                "root": root,
                "split": "valid",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
            "ds_test": {
                "root": root,
                "split": "test",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
        }
    )


@cli.command("ihdp-cov")
@click.pass_context
@click.option(
    "--root",
    type=str,
    required=True,
    help="location of dataset",
)
def ihdp_cov(
    context,
    root,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "ihdp-cov"
    experiment_dir = job_dir / dataset_name
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "root": root,
                "split": "train",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
            "ds_valid": {
                "root": root,
                "split": "valid",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
            "ds_test": {
                "root": root,
                "split": "test",
                "mode": "mu",
                "seed": context.obj.get("seed"),
            },
        }
    )


@cli.command("ensemble")
@click.pass_context
@click.option("--dim-hidden", default=200, type=int, help="num neurons")
@click.option("--dim-output", default=2, type=int, help="output dimensionality")
@click.option("--depth", default=3, type=int, help="depth of feature extractor")
@click.option(
    "--negative-slope",
    default=-1,
    type=float,
    help="negative slope of leaky relu, default=-1 use elu",
)
@click.option(
    "--dropout-rate", default=0.2, type=float, help="dropout rate, default=0.1"
)
@click.option(
    "--spectral-norm",
    default=0.95,
    type=float,
    help="Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    type=float,
    help="learning rate for gradient descent, default=1e-3",
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="number of examples to read during each training step, default=100",
)
@click.option(
    "--epochs", type=int, default=500, help="number of training epochs, default=50"
)
@click.option(
    "--ensemble-size",
    type=int,
    default=5,
    help="number of models in ensemble, default=1",
)
def ensemble(
    context,
    dim_hidden,
    dim_output,
    depth,
    negative_slope,
    dropout_rate,
    spectral_norm,
    learning_rate,
    batch_size,
    epochs,
    ensemble_size,
):
    if context.obj["tune"]:
        context.obj.update(
            {
                "dim_output": dim_output,
                "epochs": epochs,
                "ensemble_size": ensemble_size,
            }
        )
        workflows.tuning.tune_tarnet(config=context.obj)
    else:
        context.obj.update(
            {
                "dim_hidden": dim_hidden,
                "depth": depth,
                "dim_output": dim_output,
                "negative_slope": negative_slope,
                "dropout_rate": dropout_rate,
                "spectral_norm": spectral_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "ensemble_size": ensemble_size,
            }
        )

        @ray.remote(
            num_gpus=context.obj.get("gpu_per_trial"),
            num_cpus=context.obj.get("cpu_per_trial"),
        )
        def trainer(**kwargs):
            func = workflows.training.tarnet_trainer(**kwargs)
            return func

        results = []
        for trial in range(context.obj.get("num_trials")):
            for ensemble_id in range(ensemble_size):
                results.append(
                    trainer.remote(
                        config=context.obj,
                        experiment_dir=context.obj.get("experiment_dir"),
                        trial=trial,
                        ensemble_id=ensemble_id,
                    )
                )
        ray.get(results)


@cli.command("deep-kernel-gp")
@click.pass_context
@click.option("--kernel", default="Matern32", type=str, help="GP kernel")
@click.option(
    "--num-inducing-points",
    default=100,
    type=int,
    help="Number of Deep GP Inducing Points",
)
@click.option("--dim-hidden", default=200, type=int, help="num neurons")
@click.option("--dim-output", default=1, type=int, help="output dimensionality")
@click.option("--depth", default=3, type=int, help="depth of feature extractor")
@click.option(
    "--negative-slope",
    default=-1,
    type=float,
    help="negative slope of leaky relu, default=-1 use elu",
)
@click.option(
    "--dropout-rate", default=0.1, type=float, help="dropout rate, default=0.2"
)
@click.option(
    "--spectral-norm",
    default=0.95,
    type=float,
    help="Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    type=float,
    help="learning rate for gradient descent, default=1e-3",
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="number of examples to read during each training step, default=100",
)
@click.option(
    "--epochs", type=int, default=1000, help="number of training epochs, default=500"
)
def deep_kernel_gp(
    context,
    kernel,
    num_inducing_points,
    dim_hidden,
    dim_output,
    depth,
    negative_slope,
    dropout_rate,
    spectral_norm,
    learning_rate,
    batch_size,
    epochs,
):
    if context.obj["tune"]:
        context.obj.update(
            {
                "epochs": epochs,
                "dim_output": dim_output,
            }
        )
        workflows.tuning.tune_deep_kernel_gp(config=context.obj)
    else:
        context.obj.update(
            {
                "kernel": kernel,
                "num_inducing_points": num_inducing_points,
                "dim_hidden": dim_hidden,
                "depth": depth,
                "dim_output": dim_output,
                "negative_slope": negative_slope,
                "dropout_rate": dropout_rate,
                "spectral_norm": spectral_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
            }
        )

        @ray.remote(
            num_gpus=context.obj.get("gpu_per_trial"),
            num_cpus=context.obj.get("cpu_per_trial"),
        )
        def trainer(**kwargs):
            func = workflows.training.deep_kernel_gp_trainer(**kwargs)
            return func

        results = []
        for trial in range(context.obj.get("num_trials")):
            results.append(
                trainer.remote(
                    config=context.obj,
                    experiment_dir=context.obj.get("experiment_dir"),
                    trial=trial,
                )
            )
        ray.get(results)


if __name__ == "__main__":
    cli()
