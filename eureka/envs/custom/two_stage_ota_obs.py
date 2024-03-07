class TwoStageOTA(environment.Environment):
    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x0, state.x1, state.x2, state.x3, state.x4, state.x5, state.x6, state.x7, state.x8, state.x9, state.x10, state.x11, state.x12, state.x13, state.x14, state.x15]).squeeze()
