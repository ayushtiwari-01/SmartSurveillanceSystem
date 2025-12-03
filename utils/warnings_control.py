import warnings


def suppress_pkg_resources_warnings() -> None:
    """Temporarily suppress noisy third-party warnings at runtime.

    - pkg_resources deprecation notices
    - torch.load weights_only FutureWarning emitted by dependencies
    """
    # pkg_resources deprecation warning
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    # torch.load weights_only future warning from third-party packages
    warnings.filterwarnings(
        "ignore",
        message=".*You are using `torch.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
