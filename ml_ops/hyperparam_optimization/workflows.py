"""Checkpoint-aware Temporal workflows for BERT training, evaluation, and sweeps.

This module layers several workflows on top of each other to keep the example
both realistic and tutorial-friendly:

- ``CheckpointedBertTrainingWorkflow`` owns a single fine-tuning run, including
  dataset snapshot materialization and checkpoint-aware restarts.
- ``BertEvalWorkflow`` evaluates a completed run on a public dataset split.
- ``CoordinatorWorkflow`` wires training + evaluation together for one or more
  configurations so that callers see a simple "run config → eval metrics" API.
- ``SweepWorkflow`` and ``LadderSweepWorkflow`` explore the hyperparameter
  space by calling the coordinator many times, either with random sampling or
  a ladder-style, TPE-inspired schedule.

The heavy ML work (tokenization, model training, evaluation) is delegated to
activities in ``bert_activities.py``; the workflows here remain deterministic
and focus purely on orchestration.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.workflows.train_tune.bert_sweeps.custom_types import (
        BertEvalRequest,
        BertEvalResult,
        BertFineTuneConfig,
        BertFineTuneRequest,
        BertFineTuneResult,
        CheckpointInfo,
        CoordinatorWorkflowConfig,
        CoordinatorWorkflowInput,
        DatasetSnapshotRequest,
        DatasetSnapshotResult,
        SweepRequest,
        SweepResult,
        SweepSpace,
        TrialResult,
    )


# ----------------------------------------------------------------------------------
# Checkpointed BERT Training Workflow
# ----------------------------------------------------------------------------------
@workflow.defn
class CheckpointedBertTrainingWorkflow:
    """Workflow that runs checkpoint-aware fine-tuning with a shared snapshot.

    The pattern here is:

    1. Materialize (or reuse) a dataset snapshot so that training becomes
       reproducible across workers and retries.
    2. Run a single long-lived training activity that periodically saves model
       checkpoints and reports progress through signals.
    3. Expose lightweight queries so external clients can inspect the most
       recent checkpoint while the run is still in flight.
    """

    def __init__(self) -> None:
        self.latest_checkpoint: CheckpointInfo | None = None
        self.run_id = None

    @workflow.signal
    def update_checkpoint(self, info: CheckpointInfo) -> None:
        """Record the most recent checkpoint information in workflow state.

        Activities call this signal whenever a new checkpoint is persisted so
        that the workflow can expose it via :py:meth:`get_latest_checkpoint`.
        """
        self.latest_checkpoint = info

    @workflow.query
    def get_latest_checkpoint(self) -> CheckpointInfo | None:
        """Expose the most recently recorded checkpoint (if any)."""
        return self.latest_checkpoint

    # Main Workflow Function
    @workflow.run
    async def run(self, config: BertFineTuneConfig) -> BertFineTuneResult:
        """Run a single checkpoint-aware fine-tuning job."""
        # Prefer a coordinator-provided run_id when present so that the
        # coordinator can control how training/eval artifacts are named and the
        # evaluation workflow can later find the right checkpoint directory.
        if config.run_id:
            run_id = config.run_id
        else:
            # Fallback for direct usage without a coordinator: derive a
            # human-friendly, unique run identifier from Temporal's run_id.
            run_id = f"bert-checkpointed-{workflow.info().run_id}"
            config.run_id = run_id

        self.run_id = run_id

        workflow.logger.info(
            "Starting checkpointed BERT run for model %s on %s/%s",
            config.model_name,
            config.dataset_name,
            config.dataset_config_name,
        )

        # Step 1: Materialize (or reuse) a dataset snapshot for this configuration.
        snapshot_request = DatasetSnapshotRequest(
            run_id=run_id,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config_name,
            max_samples=config.max_train_samples,
        )

        snapshot: DatasetSnapshotResult = await workflow.execute_activity(
            "create_dataset_snapshot",
            snapshot_request,
            start_to_close_timeout=timedelta(minutes=10),
        )

        # Step 2: Run checkpoint-aware fine-tuning, optionally resuming from the
        # latest known checkpoint path (if one has already been recorded).
        resume_from = self.latest_checkpoint.path if self.latest_checkpoint else None

        request = BertFineTuneRequest(
            run_id=run_id,
            config=config,
            dataset_snapshot=snapshot,
            resume_from_checkpoint=resume_from,
        )

        result: BertFineTuneResult | dict = await workflow.execute_activity(
            "fine_tune_bert",
            request,
            start_to_close_timeout=timedelta(hours=2),
        )

        # Guard against cases where the Pydantic data converter returns a plain
        # dict instead of a model instance (e.g., if imports are misaligned).

        if isinstance(result, dict):
            run_id = result.get("run_id")
            checkpoints_saved = result.get("total_checkpoints_saved")

        else:
            run_id = result.run_id
            checkpoints_saved = result.total_checkpoints_saved

        workflow.logger.info(
            "Completed checkpointed BERT run %s (checkpoints_saved=%s)",
            run_id,
            checkpoints_saved,
        )

        # If we got a dict back, re-wrap it as a BertFineTuneResult so callers
        # see a consistent type.
        if isinstance(result, dict):
            return BertFineTuneResult(**result)
        return result


# ----------------------------------------------------------------------------------
# BERT Evaluation Workflow
# ----------------------------------------------------------------------------------
@workflow.defn
class BertEvalWorkflow:
    """Workflow that evaluates a fine-tuned BERT model on a public dataset."""

    # Main Workflow Function
    @workflow.run
    async def run(self, input: BertEvalRequest) -> BertEvalResult:
        """Execute evaluation for a fine-tuned BERT run."""
        if isinstance(input, dict):
            run_id = input.get("run_id")
            dataset_name = input.get("dataset_name")
            dataset_config_name = input.get("dataset_config_name")
            split = input.get("split")
        else:
            run_id = input.run_id
            dataset_name = input.dataset_name
            dataset_config_name = input.dataset_config_name
            split = input.split

        workflow.logger.info(
            "Starting BERT evaluation workflow for run %s on %s/%s[%s]",
            run_id,
            dataset_name,
            dataset_config_name,
            split,
        )

        result: BertEvalResult | dict = await workflow.execute_activity(
            "evaluate_bert_model",
            input,
            start_to_close_timeout=timedelta(minutes=10),
        )

        if isinstance(result, dict):
            out = BertEvalResult(**result)
        else:
            out = result

        workflow.logger.info(
            "Completed BERT evaluation workflow for run %s: accuracy=%.3f over %s examples",
            run_id,
            out.accuracy,
            out.num_examples,
        )

        return out


# ----------------------------------------------------------------------------------
# Parallel Run Coordination Workflow (No Sweeping, No Ladder)
# ----------------------------------------------------------------------------------
@workflow.defn
class CoordinatorWorkflow:
    """Coordinate checkpointed training and evaluation for one or more configs.

    From a caller's perspective this workflow turns a list of
    :class:`CoordinatorWorkflowConfig` objects into a list of
    :class:`BertEvalResult` objects. Internally it:

    - Normalizes and propagates a single ``run_id`` across training/eval
      configs so all artifacts live under ``./bert_runs/{run_id}``.
    - Starts a child :class:`CheckpointedBertTrainingWorkflow` per config.
    - Once all training runs are finished, starts a matching
      :class:`BertEvalWorkflow` per config and returns the results.
    """

    def __init__(self):
        self.run_ids = []
        self.run_pointers = []
        self.eval_pointers = []

    def set_run_id(self, cfg: CoordinatorWorkflowConfig) -> None:
        """Choose and propagate a canonical ``run_id`` for a config.

        The same logical ``run_id`` is written into the high-level config, the
        nested fine-tuning config, and the evaluation config so that downstream
        activities can locate checkpoints and logs by directory alone.
        """
        # Normalize run_id across the coordinator-level config, training config,
        # and evaluation config so that a single logical identifier flows
        # through training and evaluation.
        if cfg.run_id:
            canonical_run_id = cfg.run_id
        elif cfg.fine_tune_config.run_id:
            canonical_run_id = cfg.fine_tune_config.run_id
        elif cfg.evaluation_config.run_id:
            canonical_run_id = cfg.evaluation_config.run_id
        else:
            workflow.logger.info("No run id provided, generating a new one.")
            canonical_run_id = str(workflow.uuid4())

        cfg.run_id = canonical_run_id
        cfg.fine_tune_config.run_id = canonical_run_id
        cfg.evaluation_config.run_id = canonical_run_id
        self.run_ids.append(canonical_run_id)

        # If the caller did not explicitly choose a model_path for evaluation,
        # default it to the run-scoped directory that training writes to. This
        # keeps all path decisions centralized in the coordinator.
        if cfg.evaluation_config.model_path is None:
            cfg.evaluation_config.model_path = f"./bert_runs/{canonical_run_id}"

    @workflow.run
    async def run(self, input: CoordinatorWorkflowInput) -> list[BertEvalResult]:
        """Execute the coordinator workflow and return per-config evaluation results."""
        workflow.logger.info("Coordinator workflow started")

        for config in input.configs:
            # Step 1: assign a canonical run_id and make sure the evaluation
            # config will look for checkpoints in the right location.
            self.set_run_id(cfg=config)

            # Step 2: start a checkpoint-aware training workflow as a child
            # workflow. We pass in a fresh ``BertFineTuneConfig`` so this
            # workflow remains decoupled from how callers construct configs.
            run = workflow.execute_child_workflow(
                CheckpointedBertTrainingWorkflow.run,
                BertFineTuneConfig(
                    run_id=config.run_id,
                    model_name=config.fine_tune_config.model_name,
                    dataset_name=config.fine_tune_config.dataset_name,
                    dataset_config_name=config.fine_tune_config.dataset_config_name,
                    num_epochs=config.fine_tune_config.num_epochs,
                    batch_size=config.fine_tune_config.batch_size,
                    learning_rate=config.fine_tune_config.learning_rate,
                    max_seq_length=config.fine_tune_config.max_seq_length,
                    use_gpu=bool(config.fine_tune_config.use_gpu),
                    max_train_samples=config.fine_tune_config.max_train_samples,
                    max_eval_samples=config.fine_tune_config.max_eval_samples,
                    seed=config.fine_tune_config.seed,
                ),
                id=f"checkpointed-bert-training-workflow-{config.run_id}",
                task_queue="bert-training-task-queue",
            )
            self.run_pointers.append(run)

        # Wait for all training runs to complete before starting evaluation.
        await asyncio.gather(*self.run_pointers)

        # Step 3: fan out evaluation workflows, one per training run, using the
        # run-scoped model_path that ``CheckpointedBertTrainingWorkflow`` wrote.
        for config in input.configs:
            eval_pointer = workflow.execute_child_workflow(
                BertEvalWorkflow.run,
                BertEvalRequest(
                    run_id=config.run_id,
                    dataset_name=config.evaluation_config.dataset_name,
                    dataset_config_name=config.evaluation_config.dataset_config_name,
                    split=config.evaluation_config.split,
                    max_eval_samples=config.fine_tune_config.max_eval_samples,
                    max_seq_length=config.evaluation_config.max_seq_length,
                    batch_size=config.evaluation_config.batch_size,
                    use_gpu=bool(config.evaluation_config.use_gpu),
                    model_path=config.evaluation_config.model_path,
                    seed=config.evaluation_config.seed,
                ),
                id=f"bert-eval-workflow-{config.run_id}",
            )
            self.eval_pointers.append(eval_pointer)
        results = await asyncio.gather(*self.eval_pointers)
        return results


# ----------------------------------------------------------------------------------
# Sweep Workflow (No Ladder)
# ----------------------------------------------------------------------------------
@workflow.defn
class SweepWorkflow:
    """Random hyperparameter sweep over a small search space.

    This workflow is intentionally simple: it samples configurations uniformly
    from :class:`SweepSpace`, runs a child :class:`CoordinatorWorkflow` for
    each trial, and returns a sorted leaderboard so you can inspect which
    settings worked best.
    """

    @workflow.run
    async def run(self, req: SweepRequest) -> list[BertEvalResult]:
        rng = workflow.random()
        trial_cfgs: list[CoordinatorWorkflowConfig] = []
        for i in range(req.num_trials):
            cfg = req.base.model_copy(deep=True)

            # Deterministic run_id per trial
            run_id = f"{req.experiment_id}-{i:04d}"
            cfg.run_id = run_id

            # Sample hyperparams
            selected_batch_size = rng.choice(req.space.batch_size)
            cfg.fine_tune_config.batch_size = selected_batch_size
            cfg.evaluation_config.batch_size = selected_batch_size
            workflow.logger.info("Trial %s: sampled batch_size=%s", i, selected_batch_size)

            selected_max_seq_length = rng.choice(req.space.max_seq_length)
            cfg.fine_tune_config.max_seq_length = selected_max_seq_length
            cfg.evaluation_config.max_seq_length = selected_max_seq_length
            workflow.logger.info("Trial %s: sampled max_seq_length=%s", i, selected_max_seq_length)

            selected_epochs = rng.choice(req.space.num_epochs)
            cfg.fine_tune_config.num_epochs = selected_epochs
            workflow.logger.info("Trial %s: sampled num_epochs=%s", i, selected_epochs)

            # log-uniform lr
            lo, hi = req.space.learning_rate
            # sample
            u = rng.random()
            cfg.fine_tune_config.learning_rate = float(
                math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            )
            workflow.logger.info(
                "Trial %s: sampled learning_rate=%.6f", i, cfg.fine_tune_config.learning_rate
            )
            trial_cfgs.append(cfg)

        eval_results: list[BertEvalResult] = await workflow.execute_child_workflow(
            CoordinatorWorkflow.run,
            CoordinatorWorkflowInput(configs=trial_cfgs),
            id=f"coordinator-{req.experiment_id}",
        )

        trials: list[TrialResult] = []
        by_run = {r.run_id: r for r in eval_results}

        for cfg in trial_cfgs:
            er = by_run[cfg.run_id]
            score = er.accuracy
            trials.append(
                TrialResult(
                    run_id=cfg.run_id,
                    fine_tune_config=cfg.fine_tune_config,
                    eval_result=er,
                    score=score,
                )
            )

        trials.sort(key=lambda t: t.score, reverse=True)
        best = trials[0]

        return SweepResult(
            experiment_id=req.experiment_id,
            metric="accuracy",
            best_run_id=best.run_id,
            best_score=best.score,
            leaderboard=trials,
        )


@dataclass
class _TrialObs:
    run_id: str
    cfg: CoordinatorWorkflowConfig
    stage_idx: int
    score: float
    """Lightweight observation used to fit TPE-style proposal distributions."""


# ----------------------------------------------------------------------------------
# Scaling Ladder Workflow
# ----------------------------------------------------------------------------------
@workflow.defn
class LadderSweepWorkflow:
    """Ladder-style hyperparameter sweep using a simple TPE-inspired sampler.

    Instead of giving every configuration the full training budget up front,
    the ladder schedule:

    - Starts many cheap, low-epoch trials on a subset of the data.
    - Promotes only the best trials to higher "rungs" with more epochs / data.
    - Continually proposes new candidates using :meth:`_tpe_suggest`, which
      biases sampling toward regions that have historically worked well.

    This pattern lets you explore more of the search space on a fixed compute
    budget while still remaining fully deterministic from Temporal's point of
    view (all randomness flows through ``workflow.random()``).
    """

    @staticmethod
    def _clip(p: float, eps: float = 1e-9) -> float:
        """Clamp a probability into a numerically-safe open interval."""
        return max(eps, min(1.0 - eps, p))

    @staticmethod
    async def _run_one_cfg(
        sem: asyncio.Semaphore, cfg: CoordinatorWorkflowConfig, stage_label: str
    ) -> BertEvalResult:
        """Run a single coordinator config, respecting the shared semaphore."""
        async with sem:
            results: list[BertEvalResult] = await workflow.execute_child_workflow(
                CoordinatorWorkflow.run,
                CoordinatorWorkflowInput(configs=[cfg]),
                id=f"coordinator-{cfg.run_id}-{stage_label}",
            )
            return results[0]

    @staticmethod
    def _sample_categorical_weighted(rng, choices: list[int], weights: dict[int, float]) -> int:
        """Sample from ``choices`` using (possibly unnormalized) non-negative weights."""
        total = sum(max(0.0, weights.get(c, 0.0)) for c in choices)
        if total <= 0:
            return rng.choice(choices)

        r = rng.random() * total
        acc = 0.0
        for c in choices:
            acc += max(0.0, weights.get(c, 0.0))
            if r <= acc:
                return c
        return choices[-1]

    @staticmethod
    def _fit_log_normal_params(values: list[float]) -> tuple[float, float]:
        """Fit (mu, sigma) of a log-normal distribution to positive values."""
        xs = [math.log(v) for v in values if v > 0]
        if not xs:
            return (0.0, 1.0)
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
        sigma = math.sqrt(max(var, 1e-6))
        return (mu, sigma)

    @staticmethod
    def _normal_pdf(x: float, mu: float, sigma: float) -> float:
        """Evaluate a univariate normal PDF; used by the TPE scorer."""
        z = (x - mu) / sigma
        return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * z * z)

    @staticmethod
    def cat_weights(obs: list[_TrialObs], attr: str, choices: list[int]) -> dict[int, float]:
        """Count how often each categorical value appears, with Laplace smoothing."""
        c = Counter(getattr(o.cfg.fine_tune_config, attr) for o in obs)
        return {v: float(c.get(v, 0) + 1) for v in choices}  # Laplace smoothing

    @staticmethod
    def _tpe_suggest(
        rng,
        history: list[_TrialObs],
        base: CoordinatorWorkflowConfig,
        space: SweepSpace,
        *,
        gamma: float = 0.25,
        n_candidates: int = 32,
        epsilon_random: float = 0.30,
    ) -> CoordinatorWorkflowConfig:
        """Propose a new configuration using a TPE-style density ratio.

        The sampler splits historical trials into a "good" set (top ``gamma``
        fraction) and a "bad" set, builds simple distributions for each, and
        then scores candidate hyperparameters by how much more likely they are
        under the good distribution than the bad one.
        """
        if len(history) < 8 or rng.random() < epsilon_random:
            cfg = base.model_copy(deep=True)

            selected_batch_size = rng.choice(space.batch_size)
            cfg.fine_tune_config.batch_size = selected_batch_size
            cfg.evaluation_config.batch_size = selected_batch_size
            workflow.logger.info("Random TPE trial: sampled batch_size=%s", selected_batch_size)

            selected_max_seq_length = rng.choice(space.max_seq_length)
            cfg.fine_tune_config.max_seq_length = selected_max_seq_length
            cfg.evaluation_config.max_seq_length = selected_max_seq_length
            workflow.logger.info(
                "Random TPE trial: sampled max_seq_length=%s", selected_max_seq_length
            )

            selected_epochs = rng.choice(space.num_epochs)
            cfg.fine_tune_config.num_epochs = selected_epochs
            workflow.logger.info("Random TPE trial: sampled num_epochs=%s", selected_epochs)

            lo, hi = space.learning_rate
            u = rng.random()
            cfg.fine_tune_config.learning_rate = float(
                math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            )
            workflow.logger.info(
                "Random TPE trial: sampled learning_rate=%.6f", cfg.fine_tune_config.learning_rate
            )
            return cfg

        sorted_hist = sorted(history, key=lambda o: o.score, reverse=True)
        n_good = max(1, int(math.ceil(gamma * len(sorted_hist))))
        good = sorted_hist[:n_good]
        bad = sorted_hist[n_good:] if n_good < len(sorted_hist) else sorted_hist[-1:]
        good_bs_w = LadderSweepWorkflow.cat_weights(good, "batch_size", space.batch_size)
        bad_bs_w = LadderSweepWorkflow.cat_weights(bad, "batch_size", space.batch_size)
        good_ms_w = LadderSweepWorkflow.cat_weights(good, "max_seq_length", space.max_seq_length)
        bad_ms_w = LadderSweepWorkflow.cat_weights(bad, "max_seq_length", space.max_seq_length)
        good_lrs = [o.cfg.fine_tune_config.learning_rate for o in good]
        bad_lrs = [o.cfg.fine_tune_config.learning_rate for o in bad]
        g_mu, g_sig = LadderSweepWorkflow._fit_log_normal_params(good_lrs)
        b_mu, b_sig = LadderSweepWorkflow._fit_log_normal_params(bad_lrs)
        lo, hi = space.learning_rate
        log_lo, log_hi = math.log(lo), math.log(hi)

        def score_candidate(bs: int, ms: int, lr: float) -> float:
            pg_bs = good_bs_w.get(bs, 1.0)
            pb_bs = bad_bs_w.get(bs, 1.0)
            pg_ms = good_ms_w.get(ms, 1.0)
            pb_ms = bad_ms_w.get(ms, 1.0)

            x = math.log(lr)
            pg_lr = LadderSweepWorkflow._normal_pdf(x, g_mu, g_sig)
            pb_lr = LadderSweepWorkflow._normal_pdf(x, b_mu, b_sig)

            return (pg_bs / pb_bs) * (pg_ms / pb_ms) * ((pg_lr + 1e-12) / (pb_lr + 1e-12))

        best = None
        best_ratio = -1.0

        for _ in range(n_candidates):
            bs = LadderSweepWorkflow._sample_categorical_weighted(rng, space.batch_size, good_bs_w)
            ms = LadderSweepWorkflow._sample_categorical_weighted(
                rng, space.max_seq_length, good_ms_w
            )

            # Box-Muller using rng (deterministic)
            u1 = LadderSweepWorkflow._clip(rng.random())
            u2 = LadderSweepWorkflow._clip(rng.random())
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

            x = g_mu + g_sig * z
            x = max(log_lo, min(log_hi, x))
            lr = float(math.exp(x))

            r = score_candidate(bs, ms, lr)
            if r > best_ratio:
                best_ratio = r
                best = (bs, ms, lr)

        cfg = base.model_copy(deep=True)
        cfg.fine_tune_config.batch_size = best[0]
        cfg.evaluation_config.batch_size = best[0]
        cfg.fine_tune_config.max_seq_length = best[1]
        cfg.evaluation_config.max_seq_length = best[1]
        cfg.fine_tune_config.learning_rate = best[2]
        return cfg

    @workflow.run
    async def run(self, req: SweepRequest) -> list[BertEvalResult]:
        """Run a full ladder sweep and return the final rung's leaderboard.

        The ladder is encoded as a list of tuples:
        ``(epochs, max_train_samples, keep_top_k, n_new_candidates)``.
        Each rung promotes the best ``keep_top_k`` survivors from the previous
        rung, optionally adds ``n_new_candidates`` TPE-proposed configs, and
        then runs a coordinating workflow for every candidate at that budget.
        """
        rng = workflow.random()
        sem = asyncio.Semaphore(req.max_concurrency)
        history: list[_TrialObs] = []

        # (epochs, max_train_samples, keep_top_k, n_new_candidates)
        # NOTE: n_new must be present for every stage; last stage uses 0 new candidates.
        stages: list[tuple[int, int | None, int, int]] = [
            (1, 1000, max(1, req.num_trials // 4), req.num_trials),
            (2, 2000, max(1, req.num_trials // 6), max(1, req.num_trials // 2)),
            (3, 5000, max(1, req.num_trials // 8), max(1, req.num_trials // 4)),
            # Final rung: evaluate remaining survivors at full budget; no new candidates
            (4, req.base.fine_tune_config.max_train_samples, 1, 0),
        ]

        survivors: list[CoordinatorWorkflowConfig] = []
        last_ranked: list[BertEvalResult] = []

        for stage_idx, (epochs, max_train, keep_k, n_new) in enumerate(stages):
            stage_cfgs: list[CoordinatorWorkflowConfig] = []

            # Promote survivors (same run_id; higher budget)
            for cfg in survivors:
                c = cfg.model_copy(deep=True)
                c.fine_tune_config.num_epochs = epochs
                c.fine_tune_config.max_train_samples = max_train

                # Keep evaluation aligned (in case something drifted between rungs).
                c.evaluation_config.batch_size = c.fine_tune_config.batch_size
                c.evaluation_config.max_seq_length = c.fine_tune_config.max_seq_length

                # Ensure model_path exists for eval
                if c.evaluation_config.model_path is None and c.run_id:
                    c.evaluation_config.model_path = f"./bert_runs/{c.run_id}"

                stage_cfgs.append(c)

            # Add new candidates proposed by TPE (continual HPO)
            for j in range(n_new):
                # ``_tpe_suggest`` clones the base config and mutates
                # hyperparameters based on the accumulated history.
                c = LadderSweepWorkflow._tpe_suggest(rng, history, req.base, req.space)

                run_id = f"{req.experiment_id}-s{stage_idx}-{j:04d}"
                c.run_id = run_id
                c.fine_tune_config.run_id = run_id
                c.evaluation_config.run_id = run_id

                # Keep eval config consistent with train config (belt + suspenders)
                c.evaluation_config.batch_size = c.fine_tune_config.batch_size
                c.evaluation_config.max_seq_length = c.fine_tune_config.max_seq_length

                if c.evaluation_config.model_path is None:
                    c.evaluation_config.model_path = f"./bert_runs/{run_id}"

                c.fine_tune_config.num_epochs = epochs
                c.fine_tune_config.max_train_samples = max_train
                seed = await workflow.execute_activity(
                    "set_seed",
                    req.seed,
                    start_to_close_timeout=timedelta(seconds=10),
                )
                c.fine_tune_config.seed = seed
                c.evaluation_config.seed = seed

                stage_cfgs.append(c)

            # If we have no work this stage, stop cleanly
            if not stage_cfgs:
                workflow.logger.warning("Stage %s produced no configs; stopping.", stage_idx)
                break

            workflow.logger.info(
                "Ladder stage %s: trials=%s (survivors=%s, new=%s) epochs=%s max_train=%s keep_k=%s",
                stage_idx,
                len(stage_cfgs),
                len(survivors),
                n_new,
                epochs,
                max_train,
                keep_k,
            )

            stage_results: list[BertEvalResult] = await workflow.execute_child_workflow(
                CoordinatorWorkflow.run,
                CoordinatorWorkflowInput(configs=stage_cfgs),
                id=f"coordinator-{req.experiment_id}-stage-{stage_idx}",
            )

            # Rank by accuracy for logging and survivor selection.
            last_ranked = sorted(stage_results, key=lambda r: r.accuracy, reverse=True)

            # Update history used by TPE with observations from this rung.
            by_run = {r.run_id: r for r in stage_results}
            for cfg in stage_cfgs:
                r = by_run[cfg.run_id]
                history.append(
                    _TrialObs(run_id=cfg.run_id, cfg=cfg, stage_idx=stage_idx, score=r.accuracy)
                )

            # Select survivors for the next rung.
            top = last_ranked[:keep_k]
            top_ids = {r.run_id for r in top}
            survivors = [cfg for cfg in stage_cfgs if cfg.run_id in top_ids]

            workflow.logger.info(
                "Stage %s top: %s",
                stage_idx,
                [(r.run_id, r.accuracy) for r in top[: min(5, len(top))]],
            )

            # Early exit only if (a) we are at the final rung. In that case
            # we run an ablation-style baseline on the best configuration and
            # annotate the best result with its improvement over that baseline.
            is_last_stage = stage_idx == len(stages) - 1
            if is_last_stage:
                if not last_ranked:
                    return last_ranked

                best_result = last_ranked[0]
                best_run_id = best_result.run_id

                best_cfg = next((cfg for cfg in stage_cfgs if cfg.run_id == best_run_id), None)
                if best_cfg is None:
                    workflow.logger.warning(
                        "Could not locate config for best run %s; skipping ablation.",
                        best_run_id,
                    )
                    return last_ranked

                # Build an ablation config by reusing the best hyperparameters
                # but reverting the training budget (epochs / max_train_samples)
                # to the first ladder stage. This provides a baseline to
                # quantify how much the full-ladder schedule improved accuracy.
                ablation_cfg = best_cfg.model_copy(deep=True)
                ablation_run_id = f"{best_cfg.run_id}-ablation"
                ablation_cfg.run_id = ablation_run_id
                ablation_cfg.fine_tune_config.run_id = ablation_run_id
                ablation_cfg.evaluation_config.run_id = ablation_run_id

                base_epochs, base_max_train, _keep0, _nnew0 = stages[0]
                ablation_cfg.fine_tune_config.num_epochs = base_epochs
                ablation_cfg.fine_tune_config.max_train_samples = base_max_train

                ablation_cfg.evaluation_config.batch_size = ablation_cfg.fine_tune_config.batch_size
                ablation_cfg.evaluation_config.max_seq_length = (
                    ablation_cfg.fine_tune_config.max_seq_length
                )

                old_default = f"./bert_runs/{best_cfg.run_id}"
                new_default = f"./bert_runs/{ablation_run_id}"

                if ablation_cfg.evaluation_config.model_path in (None, old_default):
                    ablation_cfg.evaluation_config.model_path = new_default

                try:
                    seed = await workflow.execute_activity(
                        "set_seed",
                        req.seed,
                        start_to_close_timeout=timedelta(seconds=10),
                    )
                    ablation_cfg.fine_tune_config.seed = seed
                    ablation_cfg.evaluation_config.seed = seed

                    ablation_result = await LadderSweepWorkflow._run_one_cfg(
                        sem, ablation_cfg, "ablation"
                    )

                    best_result.baseline_accuracy = ablation_result.accuracy
                    best_result.improvement_vs_baseline = (
                        best_result.accuracy - ablation_result.accuracy
                    )
                    workflow.logger.info(
                        "Ablation baseline for best run %s: baseline_accuracy=%.3f, "
                        "best_accuracy=%.3f, improvement=%.3f",
                        best_run_id,
                        ablation_result.accuracy,
                        best_result.accuracy,
                        best_result.improvement_vs_baseline,
                    )
                except Exception:
                    # If the ablation run fails for any reason, we still return
                    # the final rung leaderboard; the metadata fields remain unset.
                    workflow.logger.warning(
                        "Ablation run for best configuration %s failed; "
                        "returning final rung leaderboard without ablation metadata.",
                        best_run_id,
                        exc_info=True,
                    )

                return last_ranked

            # Otherwise keep iterating unless we're down to <= 1 survivor,
            # in which case additional rungs would not change the ordering.
            if len(survivors) <= 1:
                continue

        # Fallback: return best seen overall if stages exhausted unexpectedly
        if last_ranked:
            return last_ranked

        best_overall = sorted(history, key=lambda o: o.score, reverse=True)
        if not best_overall:
            return []

        best_cfg = best_overall[0].cfg.model_copy(deep=True)
        if best_cfg.run_id is None:
            best_cfg.run_id = best_overall[0].run_id
        best_cfg.fine_tune_config.run_id = best_cfg.run_id
        best_cfg.evaluation_config.run_id = best_cfg.run_id
        best_cfg.evaluation_config.batch_size = best_cfg.fine_tune_config.batch_size
        best_cfg.evaluation_config.max_seq_length = best_cfg.fine_tune_config.max_seq_length

        old_default = f"./bert_runs/{best_cfg.run_id}"
        new_default = f"./bert_runs/{ablation_run_id}"

        if best_cfg.evaluation_config.model_path in (None. old_default):
            best_cfg.evaluation_config.model_path = new_default
        best_result = await LadderSweepWorkflow._run_one_cfg(sem, best_cfg, "best-fallback")

        # Best-effort ablation in the fallback path as well: use the same
        # hyperparameters but revert training budget to the first stage.
        try:
            ablation_cfg = best_cfg.model_copy(deep=True)
            ablation_run_id = f"{best_cfg.run_id}-ablation"
            ablation_cfg.run_id = ablation_run_id
            ablation_cfg.fine_tune_config.run_id = ablation_run_id
            ablation_cfg.evaluation_config.run_id = ablation_run_id

            base_epochs, base_max_train, _keep0, _nnew0 = stages[0]
            ablation_cfg.fine_tune_config.num_epochs = base_epochs
            ablation_cfg.fine_tune_config.max_train_samples = base_max_train

            ablation_cfg.evaluation_config.batch_size = ablation_cfg.fine_tune_config.batch_size
            ablation_cfg.evaluation_config.max_seq_length = (
                ablation_cfg.fine_tune_config.max_seq_length
            )
            if ablation_cfg.evaluation_config.model_path is None:
                ablation_cfg.evaluation_config.model_path = f"./bert_runs/{ablation_run_id}"

            seed = await workflow.execute_activity(
                "set_seed",
                req.seed,
                start_to_close_timeout=timedelta(seconds=10),
            )
            ablation_cfg.fine_tune_config.seed = seed
            ablation_cfg.evaluation_config.seed = seed

            ablation_result = await LadderSweepWorkflow._run_one_cfg(sem, ablation_cfg, "ablation")
            best_result.baseline_accuracy = ablation_result.accuracy
            best_result.improvement_vs_baseline = best_result.accuracy - ablation_result.accuracy
        except Exception:
            workflow.logger.warning(
                "Fallback ablation run for best configuration %s failed; "
                "returning best result without ablation metadata.",
                best_cfg.run_id,
                exc_info=True,
            )

        return [best_result]
