"""Experiment catalog and orchestration for the laboratory."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn, optim

from src.config import AppConfig
from src.data.datamodule import UTKFaceDataModule
from src.evaluation.metrics import MultiTaskEvaluator
from src.evaluation.artifacts import ExperimentArtifactWriter
from src.evaluation.plots import ResultPlotter
from src.evaluation.reporter import ExperimentResult, ExperimentStatus
from src.models.cnn import MultiTaskCNN
from src.training.losses import MultiTaskLoss
from src.training.trainer import MultiTaskTrainer
from src.utils import set_seed
from src.models.mlp_todo import MultiTaskMLP
from src.models.resnet_todo import MultiTaskResNet
from src.baselines.classical_todo import ClassicalBaseline
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time

@dataclass(frozen=True)
class ExperimentSpec:
    """Configuration for one base experiment or one single-change ablation."""

    strategy_id: str
    strategy_name: str
    name: str
    variant: str
    changed_component: str
    implemented: bool
    model_kind: str
    use_augmentation: bool = True
    dropout: float = 0.4
    lambda_age: float = 0.01
    learning_rate: float = 1e-3


def build_experiment_catalog(config: AppConfig) -> dict[str, ExperimentSpec]:
    """Return all required strategies and their expected ablation studies.

    E3 is delivered as a complete example. E1, E2, E4 and E5 are intentionally
    visible but not implemented so students can complete and report them.
    """

    low_lambda = config.lambda_age / 10
    high_lambda = config.lambda_age * 10

    specs = [
        # E1: classical baseline. The estimator itself will use scikit-learn.
        ExperimentSpec("E1", "Baseline clasico", "classical_base", "base", "ninguno", True, "classical"),
        ExperimentSpec("E1", "Baseline clasico", "classical_pca_50", "ablacion", "PCA=50 componentes", True, "classical"),
        ExperimentSpec("E1", "Baseline clasico", "classical_pca_200", "ablacion", "PCA=200 componentes", True, "classical"),
        # E2: students must implement both the base MLP and its ablations.
        ExperimentSpec("E2", "MLP multitarea", "mlp_base", "base", "ninguno", True, "mlp"),
        ExperimentSpec("E2", "MLP multitarea", "mlp_no_dropout", "ablacion", "dropout=0.0", True, "mlp", dropout=0.0),
        ExperimentSpec("E2", "MLP multitarea", "mlp_lambda_low", "ablacion", f"lambda_age={low_lambda:g}", True, "mlp", lambda_age=low_lambda),
        ExperimentSpec("E2", "MLP multitarea", "mlp_lambda_high", "ablacion", f"lambda_age={high_lambda:g}", True, "mlp", lambda_age=high_lambda),
        # E3: complete PyTorch CNN example and one-change-at-a-time ablations.
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_base",
            "base",
            "ninguno",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_no_augmentation",
            "ablacion",
            "sin aumentacion",
            True,
            "cnn",
            use_augmentation=False,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_no_dropout",
            "ablacion",
            "dropout=0.0",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.0,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_lambda_low",
            "ablacion",
            f"lambda_age={low_lambda:g}",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=low_lambda,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_lambda_high",
            "ablacion",
            f"lambda_age={high_lambda:g}",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=high_lambda,
            learning_rate=config.learning_rate,
        ),
        # E4: frozen ResNet transfer learning exercises.
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_base", "base", "ninguno", True, "resnet_frozen"),
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_no_augmentation", "ablacion", "sin aumentacion", True, "resnet_frozen", use_augmentation=False),
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_lambda_low", "ablacion", f"lambda_age={low_lambda:g}", True, "resnet_frozen", lambda_age=low_lambda),
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_lambda_high", "ablacion", f"lambda_age={high_lambda:g}", True, "resnet_frozen", lambda_age=high_lambda),
        # E5: fine-tuning exercises.
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_base", "base", "ninguno", True, "resnet_finetuning", learning_rate=1e-4),
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_unfreeze_more", "ablacion", "mas bloques descongelados", True, "resnet_finetuning", learning_rate=1e-4),
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_lr_low", "ablacion", "learning rate menor", True, "resnet_finetuning", learning_rate=1e-5),
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_lambda_high", "ablacion", f"lambda_age={high_lambda:g}", True, "resnet_finetuning", lambda_age=high_lambda, learning_rate=1e-4),
    ]
    return {spec.name: spec for spec in specs}


class ExperimentRunner:
    """Run selected experiments and preserve report rows for every strategy."""

    def __init__(
        self,
        config: AppConfig,
        device: torch.device,
        catalog: dict[str, ExperimentSpec],
    ) -> None:
        self.config = config
        self.device = device
        self.catalog = catalog
        self.plotter = ResultPlotter(config.plots_dir)
        self.artifact_writer = ExperimentArtifactWriter(
            config.experiments_dir,
            config.reports_dir,
        )

    def run(self, selected_names: set[str]) -> list[ExperimentResult]:
        unknown = selected_names.difference(self.catalog)
        if unknown:
            raise ValueError(f"Experimentos desconocidos: {', '.join(sorted(unknown))}")

        results: list[ExperimentResult] = []
        self.artifact_writer.start_run(selected_names)
        for spec in self.catalog.values():
            if not spec.implemented:
                results.append(self._not_implemented_result(spec))
            elif spec.name not in selected_names:
                results.append(self._not_executed_result(spec))
            else:
                result = self._run_spec(spec)
                results.append(result)
                self.artifact_writer.write_result(result)
                self.artifact_writer.update_run(selected_names, results)

        for strategy_id in ("E1", "E2", "E3", "E4", "E5"):
            self.plotter.plot_ablation_comparison(results, strategy_id)
        self.plotter.plot_global_comparison(results)
        final_status = (
            "COMPLETED"
            if all(
                result.status == ExperimentStatus.COMPLETED
                for result in results
                if result.experiment_name in selected_names
            )
            else "COMPLETED_WITH_ERRORS"
        )
        self.artifact_writer.update_run(
            selected_names,
            results,
            status=final_status,
        )
        return results

    def _run_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        if spec.model_kind == "classical":
            return self._run_classical_spec(spec)
            
        print(f"\nEjecutando {spec.name}: {spec.changed_component}")
        try:
            set_seed(self.config.seed)
            data_module = UTKFaceDataModule(
                self.config,
                use_augmentation=spec.use_augmentation,
            )
            data_module.setup()

            model, model_kwargs = self._build_model(spec)
            model = model.to(self.device)
            optimizer = optim.Adam(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=spec.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            loss_function = MultiTaskLoss(lambda_age=spec.lambda_age)
            checkpoint_path = (
                self.config.checkpoints_dir / spec.name / "best_model.pt"
            )
            trainer = MultiTaskTrainer(
                model=model,
                optimizer=optimizer,
                loss_function=loss_function,
                device=self.device,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata={
                    "experiment_name": spec.name,
                    "strategy_id": spec.strategy_id,
                    "model_name": spec.model_kind,
                    "model_kwargs": model_kwargs,
                    "image_size": self.config.image_size,
                    "lambda_age": spec.lambda_age,
                },
            )
            history, training_seconds = trainer.fit(
                data_module.train_dataloader(),
                data_module.val_dataloader(),
                epochs=self.config.epochs,
            )
            trainer.load_best_checkpoint()

            evaluator = MultiTaskEvaluator(self.device)
            evaluation = evaluator.evaluate(model, data_module.test_dataloader())
            self.artifact_writer.write_history(spec.name, history)
            self.artifact_writer.write_evaluation(
                spec.name,
                evaluation,
                sample_names=data_module.test_sample_names(),
            )
            self.plotter.plot_training_history(history, spec.name)
            self.plotter.plot_confusion_matrix(evaluation, spec.name)
            self.plotter.plot_age_predictions(evaluation, spec.name)
            self.plotter.plot_age_residuals(evaluation, spec.name)
            self.plotter.plot_age_range_errors(evaluation, spec.name)

            sizes = data_module.split_sizes()
            metrics = dict(evaluation.metrics)
            metrics.update(
                {
                    "train_samples": sizes["train"],
                    "validation_samples": sizes["validation"],
                    "test_samples": sizes["test"],
                }
            )
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.COMPLETED,
                metrics=metrics,
                trainable_parameters=self._count_trainable_parameters(model),
                training_seconds=training_seconds,
                checkpoint=str(checkpoint_path),
                message="",
            )
        except Exception as error:
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.ERROR,
                message=str(error),
            )

    def _build_model(self, spec: ExperimentSpec) -> tuple[nn.Module, dict[str, float]]:
        if spec.model_kind == "cnn":
            model_kwargs = {"dropout": spec.dropout}
            return MultiTaskCNN(**model_kwargs), model_kwargs
            
        if spec.model_kind == "mlp":
            model_kwargs = {"dropout": spec.dropout, "image_size": self.config.image_size}
            return MultiTaskMLP(**model_kwargs), model_kwargs
            
        if spec.model_kind == "resnet_frozen":
            model_kwargs = {"freeze_backbone": True, "unfrozen_blocks": 0}
            return MultiTaskResNet(**model_kwargs), model_kwargs
            
        if spec.model_kind == "resnet_finetuning":
            unfrozen = 2 if "unfreeze_more" in spec.name else 1
            model_kwargs = {"freeze_backbone": True, "unfrozen_blocks": unfrozen}
            return MultiTaskResNet(**model_kwargs), model_kwargs

        raise NotImplementedError(f"No existe una fabrica para model_kind={spec.model_kind}.")

    def _get_classical_data(self, dataloader):
        X, y_g, y_a = [], [], []
        for images, genders, ages in dataloader:
            # Convert to grayscale [B, 1, 224, 224]
            gray = TF.rgb_to_grayscale(images)
            # Resize to low res [B, 1, 64, 64]
            small = F.interpolate(gray, size=(64, 64), mode='bilinear', align_corners=False)
            X.append(small.cpu().numpy())
            y_g.append(genders.cpu().numpy())
            y_a.append(ages.cpu().numpy())
        return np.concatenate(X), np.concatenate(y_g), np.concatenate(y_a)

    def _run_classical_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        print(f"\nEjecutando Baseline Clasico {spec.name}: {spec.changed_component}")
        try:
            set_seed(self.config.seed)
            data_module = UTKFaceDataModule(
                self.config,
                use_augmentation=False,
            )
            data_module.setup()
            
            # Determine PCA components based on spec variant name
            n_comp = 100
            if "pca_50" in spec.name: n_comp = 50
            if "pca_200" in spec.name: n_comp = 200
                
            model = ClassicalBaseline(n_components=n_comp)
            
            print("Extrayendo datos clasicos (downsampling)...")
            X_train, y_g_train, y_a_train = self._get_classical_data(data_module.train_dataloader())
            X_test, y_g_test, y_a_test = self._get_classical_data(data_module.test_dataloader())
            
            start_time = time.time()
            print("Entrenando pipeline PCA + Estimadores Clasicos...")
            model.fit(X_train, y_g_train, y_a_train)
            training_seconds = time.time() - start_time
            
            print("Evaluando en test...")
            gender_preds, age_preds = model.predict(X_test)
            
            from src.evaluation.metrics import MultiTaskMetrics

            evaluation = MultiTaskMetrics.calculate(
                gender_targets=torch.as_tensor(y_g_test),
                gender_predictions=torch.as_tensor(gender_preds),
                age_targets=torch.as_tensor(y_a_test),
                age_predictions=torch.as_tensor(age_preds),
            )
            metrics = dict(evaluation.metrics)

            checkpoint_path = (
                self.config.checkpoints_dir / spec.name / "best_model.joblib"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            from joblib import dump

            dump(model, checkpoint_path)
            self.artifact_writer.write_evaluation(
                spec.name,
                evaluation,
                sample_names=data_module.test_sample_names(),
            )
            self.plotter.plot_confusion_matrix(evaluation, spec.name)
            self.plotter.plot_age_predictions(evaluation, spec.name)
            self.plotter.plot_age_residuals(evaluation, spec.name)
            self.plotter.plot_age_range_errors(evaluation, spec.name)
            
            sizes = data_module.split_sizes()
            metrics.update({
                "train_samples": sizes["train"],
                "validation_samples": sizes["validation"],
                "test_samples": sizes["test"],
            })
            
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.COMPLETED,
                metrics=metrics,
                trainable_parameters=0,
                training_seconds=training_seconds,
                checkpoint=str(checkpoint_path),
                message="",
            )
        except Exception as error:
            import traceback
            traceback.print_exc()
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.ERROR,
                message=str(error),
            )

    @staticmethod
    def _count_trainable_parameters(model: nn.Module) -> int:
        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _not_implemented_result(spec: ExperimentSpec) -> ExperimentResult:
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_IMPLEMENTED,
            message="El experimento debe ser completado por los alumnos.",
        )

    @staticmethod
    def _not_executed_result(spec: ExperimentSpec) -> ExperimentResult:
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_EXECUTED,
            message="Implementado, pero no fue seleccionado en esta ejecucion.",
        )
