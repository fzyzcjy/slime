import wandb


def init_wandb_primary(args):
    if not args.use_wandb:
        return None

    if args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    wandb.init(
        entity=args.wandb_team,
        project=args.wandb_project,
        group=group,
        name=run_name,
        config=args.__dict__,
        settings=wandb.Settings(mode="shared", x_primary=True),
    )

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")

    return wandb.run.id


def init_wandb_secondary(wandb_run_id):
    if wandb_run_id is None:
        return

    wandb.init(
        id=wandb_run_id,
        settings=wandb.Settings(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        ),
    )
