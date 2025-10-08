from __future__ import annotations

from stable_baselines3 import PPO

from learner.env import FPLSeasonEOEnv
from learner.callbacks import ActionCountCallback


def train_demo(total_timesteps: int = 300_000) -> None:
    env = FPLSeasonEOEnv(
        n_npc=1000,
        n_per_pos={0: 10, 1: 20, 2: 20, 3: 20},
        horizon=8,
        beta_follow_eo=1.2,
        beta_follow_skill=0.8,
        rng_seed=42,
        include_week_in_obs=True,
    )

    # Concentrated picker is the default in env via learner.npc.npc_pick_xi

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=4096,
        batch_size=512,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
    )
    action_cb = ActionCountCallback(n_actions=env.action_space.n, log_freq=5000)
    model.learn(total_timesteps=total_timesteps, callback=action_cb)

    obs, _ = env.reset(seed=777)
    total_r = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(action))
        total_r += r
        if term or trunc:
            break
    print(
        f"Terminal reward (percentile-centered): {total_r:.3f}, my_total={info['my_total']:.1f}, field_mean_total={info['field_mean_total']:.1f}"
    )


if __name__ == "__main__":
    train_demo()


