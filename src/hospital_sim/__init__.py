"""Hospital patient-flow scheduling simulator."""

__version__ = "0.1.0"


def _register_gym_env() -> None:
    # Lets ``gymnasium.make("SchedSim-v0")`` work after ``import hospital_sim``.
    from hospital_sim.env.registration import register

    register()


_register_gym_env()
