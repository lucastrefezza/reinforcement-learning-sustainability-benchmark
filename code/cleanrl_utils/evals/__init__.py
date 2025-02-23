def dqn():
    import cleanrl.extra_implementations.dqn
    import cleanrl_utils.evals.dqn_eval

    return cleanrl.extra_implementations.dqn.QNetwork, cleanrl.extra_implementations.dqn.make_env, cleanrl_utils.evals.dqn_eval.evaluate


def dqn_atari():
    import cleanrl.dqn_atari
    import cleanrl_utils.evals.dqn_eval

    return cleanrl.dqn_atari.QNetwork, cleanrl.dqn_atari.make_env, cleanrl_utils.evals.dqn_eval.evaluate


def dqn_jax():
    import cleanrl.extra_implementations.dqn_jax
    import cleanrl_utils.evals.extra.dqn_jax_eval

    return cleanrl.extra_implementations.dqn_jax.QNetwork, cleanrl.extra_implementations.dqn_jax.make_env, cleanrl_utils.evals.extra.dqn_jax_eval.evaluate


def dqn_atari_jax():
    import cleanrl.extra_implementations.dqn_atari_jax
    import cleanrl_utils.evals.extra.dqn_jax_eval

    return cleanrl.extra_implementations.dqn_atari_jax.QNetwork, cleanrl.extra_implementations.dqn_atari_jax.make_env, cleanrl_utils.evals.extra.dqn_jax_eval.evaluate


def c51():
    import cleanrl.extra_implementations.c51
    import cleanrl_utils.evals.c51_eval

    return cleanrl.extra_implementations.c51.QNetwork, cleanrl.extra_implementations.c51.make_env, cleanrl_utils.evals.c51_eval.evaluate


def c51_atari():
    import cleanrl.c51_atari
    import cleanrl_utils.evals.c51_eval

    return cleanrl.c51_atari.QNetwork, cleanrl.c51_atari.make_env, cleanrl_utils.evals.c51_eval.evaluate


def c51_jax():
    import cleanrl.extra_implementations.c51_jax
    import cleanrl_utils.evals.extra.c51_jax_eval

    return cleanrl.extra_implementations.c51_jax.QNetwork, cleanrl.extra_implementations.c51_jax.make_env, cleanrl_utils.evals.extra.c51_jax_eval.evaluate


def c51_atari_jax():
    import cleanrl.extra_implementations.c51_atari_jax
    import cleanrl_utils.evals.extra.c51_jax_eval

    return cleanrl.extra_implementations.c51_atari_jax.QNetwork, cleanrl.extra_implementations.c51_atari_jax.make_env, cleanrl_utils.evals.extra.c51_jax_eval.evaluate


def ppo_atari_envpool_xla_jax_scan():
    import cleanrl.extra_implementations.ppo_atari_envpool_xla_jax_scan
    import cleanrl_utils.evals.extra.ppo_envpool_jax_eval

    return (
        (
            cleanrl.extra_implementations.ppo_atari_envpool_xla_jax_scan.Network,
            cleanrl.extra_implementations.ppo_atari_envpool_xla_jax_scan.Actor,
            cleanrl.extra_implementations.ppo_atari_envpool_xla_jax_scan.Critic,
        ),
        cleanrl.extra_implementations.ppo_atari_envpool_xla_jax_scan.make_env,
        cleanrl_utils.evals.extra.ppo_envpool_jax_eval.evaluate,
    )


MODELS = {
    "dqn": dqn,
    "dqn_atari": dqn_atari,
    "dqn_jax": dqn_jax,
    "dqn_atari_jax": dqn_atari_jax,
    "c51": c51,
    "c51_atari": c51_atari,
    "c51_jax": c51_jax,
    "c51_atari_jax": c51_atari_jax,
    "ppo_atari_envpool_xla_jax_scan": ppo_atari_envpool_xla_jax_scan,
}
