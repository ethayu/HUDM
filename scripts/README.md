# Scripts

Scripts are external consumers of the MWM library. They should use package
module CLIs such as `python -m mwm.training.lewm` or canonical public `mwm.*`
modules, never private helpers, retired compatibility modules, or root CLI
modules as Python imports.

- `local/`: desktop and developer smoke workflows.
- `slurm/`: cluster job launchers, submit wrappers, and pollers.
- `research/`: probes, one-off analysis helpers, and research-only launchers.

If a script needs behavior that only exists behind an underscore-prefixed helper,
promote that behavior into a public library module before importing it here.
