# MLX Swift LM skill

This repo ships an MLX Swift LM skill definition under `skills/mlx-swift-lm/` (the `skill.md`
file plus `references/`). The install folder name can be `mlx-swift-lm`, as shown below.
If your local copy lives at `skills/mlx-swift-lm`, just swap the source path in the
commands.

## Install globally (home directory)

Run these from the repo root, or replace `$(pwd)` with an absolute path.

### Claude Code

```sh
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/mlx-swift-lm" ~/.claude/skills/mlx-swift-lm
```

### Codex

```sh
mkdir -p ~/.codex/skills
ln -s "$(pwd)/skills/mlx-swift-lm" ~/.codex/skills/mlx-swift-lm
```

### Droid

```sh
mkdir -p ~/.agents/skills
ln -s "$(pwd)/skills/mlx-swift-lm" ~/.agents/skills/mlx-swift-lm
```

## Install per-project

Create a local skills folder in the project and link the skill there.

### Claude Code

```sh
mkdir -p .claude/skills
ln -s "$(pwd)/skills/mlx-swift-lm" .claude/skills/mlx-swift-lm
```

### Codex

```sh
mkdir -p .codex/skills
ln -s "$(pwd)/skills/mlx-swift-lm" .codex/skills/mlx-swift-lm
```

### Droid

```sh
mkdir -p .agents/skills
ln -s "$(pwd)/skills/mlx-swift-lm" .agents/skills/mlx-swift-lm
```

## Notes

- If your tool caches skills, restart it after installing.
- If you prefer a copy over a symlink, replace `ln -s` with `cp -R`.
