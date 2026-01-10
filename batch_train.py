#!/usr/bin/env python3
"""
Batch training CLI for sdpype.

Enables running recursive_train.py across multiple configuration files.

NORMAL MODE (new training runs):
1. Backup current params.yaml
2. Copy config file to params.yaml
3. Run recursive training
4. Move experiments folder
5. Delete source config file
6. Restore params.yaml

RESUME MODE (continue existing runs):
1. Backup current params.yaml
2. Restore experiments folder from output directory
3. Run recursive_train.py with --resume
4. Move updated experiments folder back
5. Restore params.yaml

Examples:
  # Start new runs with 2 generations
  uv run batch_train.py './configs/params/*gaussian*.yaml' --generations 2

  # Continue existing runs to reach 4 total generations
  uv run batch_train.py '*gaussian*' --resume 4
"""

import argparse
import json
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path


def find_most_advanced_checkpoint(checkpoint_dir: Path) -> tuple[str, int]:
    """
    Find the most advanced checkpoint in a directory.

    Returns:
        (model_id, last_completed_generation) or (None, -1) if no checkpoints found
    """
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))

    if not checkpoint_files:
        return None, -1

    best_checkpoint = None
    best_gen = -1
    best_model_id = None

    for cp_file in checkpoint_files:
        try:
            with cp_file.open() as f:
                cp_data = json.load(f)
                last_gen = cp_data.get('last_completed_generation', -1)
                if last_gen > best_gen:
                    best_gen = last_gen
                    best_model_id = cp_data.get('last_model_id')
                    best_checkpoint = cp_data
        except (json.JSONDecodeError, KeyError):
            continue

    return best_model_id, best_gen


def main():
    parser = argparse.ArgumentParser(
        description="Batch training CLI for sdpype",
        epilog="""Examples:
  # Start new runs
  uv run batch_train.py './configs/params/*rtvae*.yaml' --generations 20

  # Resume existing runs to 30 total generations
  uv run batch_train.py '*rtvae*' --resume 30"""
    )
    parser.add_argument(
        "pattern",
        help="Glob pattern for config files (normal mode) or output directories (resume mode)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations for recursive training (default: 20)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun all pipeline stages (passed to DVC)"
    )
    parser.add_argument(
        "--resume",
        type=int,
        metavar="GENERATIONS",
        help="Resume existing runs to reach GENERATIONS total (pattern matches output directories, not config files)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (show pipeline output and debug info)"
    )

    args = parser.parse_args()

    # Get matching files or directories depending on mode
    if args.resume:
        # Resume mode: match directories (existing output folders)
        all_matches = sorted(glob(args.pattern))
        matching_items = [p for p in all_matches if Path(p).is_dir()]

        if not matching_items:
            print(f"No directories matching pattern: {args.pattern}")
            print("In resume mode, pattern should match output directories from previous runs")
            sys.exit(1)
    else:
        # Normal mode: match config files
        matching_items = sorted(glob(args.pattern))

        if not matching_items:
            print(f"No files matching pattern: {args.pattern}")
            sys.exit(1)

    # Print dry-run info
    if args.dry_run:
        print(f"DRY-RUN MODE")
        print(f"Pattern: {args.pattern}")
        print(f"Generations: {args.generations}")
        print(f"Force: {args.force}")
        print(f"Verbose: {args.verbose}")
        if args.resume:
            print(f"Resume to: {args.resume} total generations")

        if args.resume:
            print(f"\nMatching directories ({len(matching_items)}):")
            for i, dirpath in enumerate(matching_items, 1):
                dirname = Path(dirpath).name
                dir_path = Path(dirpath)
                # Check both possible checkpoint locations
                checkpoint_dir_new = dir_path / "experiments" / "checkpoints"
                checkpoint_dir_direct = dir_path / "checkpoints"
                has_checkpoint = checkpoint_dir_new.exists() or checkpoint_dir_direct.exists()
                structure = "nested" if checkpoint_dir_new.exists() else ("direct" if checkpoint_dir_direct.exists() else "none")
                print(f"  {i}. {dirpath}")
                print(f"     Checkpoint: {'✓' if has_checkpoint else '✗'} ({structure})")

            force_flag = " --force" if args.force else ""
            verbose_flag = " --verbose" if args.verbose else ""
            print(f"\nWorkflow for each directory:")
            print(f"  1. Backup params.yaml -> params.backup.yaml")
            print(f"  2. Temporarily restore {{dir}}/experiments/ -> experiments/")
            print(f"  3. Restore params from checkpoint backup")
            print(f"  4. Run: uv run recursive_train.py --resume --generations {args.resume}{force_flag}{verbose_flag}")
            print(f"  5. Move experiments/ -> {{dir}}/experiments/")
            print(f"  6. Restore params.backup.yaml -> params.yaml")
            print(f"  7. Continue to next directory")
        else:
            print(f"\nMatching files ({len(matching_items)}):")
            for i, filepath in enumerate(matching_items, 1):
                basename = Path(filepath).stem
                print(f"  {i}. {filepath}")
                print(f"     -> Output folder: {basename}/")

            force_flag = " --force" if args.force else ""
            verbose_flag = " --verbose" if args.verbose else ""
            print(f"\nWorkflow for each file:")
            print(f"  1. Backup params.yaml -> params.backup.yaml")
            print(f"  2. Copy config file -> params.yaml")
            print(f"  3. Run: uv run recursive_train.py --generations {args.generations}{force_flag}{verbose_flag}")
            print(f"  4. Move experiments/ -> {{config_basename}}/")
            print(f"  5. Delete source config file")
            print(f"  6. Restore params.backup.yaml -> params.yaml")
            print(f"  7. Continue to next file")
        return

    # Normal execution
    params_path = Path("params.yaml")
    backup_path = Path("params.backup.yaml")
    experiments_path = Path("experiments")

    # Branch: Resume mode vs Normal mode
    if args.resume:
        # RESUME MODE: Process existing directories
        for i, output_dir in enumerate(matching_items, 1):
            output_path = Path(output_dir)

            # Support two directory structures:
            # 1. New batch_train.py: output_dir/experiments/checkpoints/
            # 2. Direct structure: output_dir/checkpoints/
            dir_experiments_path = output_path / "experiments"
            if dir_experiments_path.exists():
                # New structure with experiments/ subfolder
                checkpoint_dir = dir_experiments_path / "checkpoints"
                has_experiments_subfolder = True
            else:
                # Direct structure - data at root level
                checkpoint_dir = output_path / "checkpoints"
                dir_experiments_path = output_path
                has_experiments_subfolder = False

            print(f"\n[{i}/{len(matching_items)}] Resuming: {output_dir}")

            # Verify checkpoint exists
            if not checkpoint_dir.exists():
                print(f"  ✗ No checkpoint found in {checkpoint_dir}")
                print(f"  Skipping (not a valid training output directory)")
                continue

            # Find the most advanced checkpoint (handles multiple chains)
            best_model_id, best_gen = find_most_advanced_checkpoint(checkpoint_dir)
            if not best_model_id:
                print(f"  ✗ No valid checkpoints found in {checkpoint_dir}")
                print(f"  Skipping")
                continue

            print(f"  Found checkpoint: gen={best_gen}, resuming to gen={args.resume - 1}")

            try:
                # Step 1: Backup current params.yaml
                if params_path.exists():
                    print(f"  Backing up params.yaml -> params.backup.yaml")
                    shutil.copy2(params_path, backup_path)

                # Step 2: Temporarily restore experiments folder
                if has_experiments_subfolder:
                    # New structure: copy experiments/ subfolder
                    if experiments_path.exists():
                        print(f"  Removing existing experiments/ folder")
                        shutil.rmtree(experiments_path)
                    print(f"  Restoring {output_dir}/experiments/ -> experiments/")
                    shutil.copytree(dir_experiments_path, experiments_path)
                else:
                    # Direct structure: copy data directories to experiments/
                    if experiments_path.exists():
                        print(f"  Removing existing experiments/ folder")
                        shutil.rmtree(experiments_path)
                    print(f"  Restoring {output_dir}/* -> experiments/")
                    experiments_path.mkdir(exist_ok=True)
                    for subdir in ["checkpoints", "data", "metrics", "models"]:
                        src = output_path / subdir
                        if src.exists():
                            shutil.copytree(src, experiments_path / subdir)

                # Step 3: Run recursive training with resume
                cmd = ["uv", "run", "recursive_train.py", "--resume-from", best_model_id, "--generations", str(args.resume)]
                if args.force:
                    cmd.append("--force")
                if args.verbose:
                    cmd.append("--verbose")
                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True)

                # Step 4: Move updated experiments folder back
                if has_experiments_subfolder:
                    # New structure: move experiments/ back
                    if experiments_path.exists():
                        print(f"  Removing old {output_dir}/experiments/")
                        shutil.rmtree(dir_experiments_path)
                        print(f"  Moving experiments/ -> {output_dir}/experiments/")
                        shutil.move(str(experiments_path), str(dir_experiments_path))
                else:
                    # Direct structure: copy subdirs back to root
                    if experiments_path.exists():
                        print(f"  Updating {output_dir}/ with new data")
                        for subdir in ["checkpoints", "data", "metrics", "models"]:
                            src = experiments_path / subdir
                            dst = output_path / subdir
                            if src.exists():
                                if dst.exists():
                                    shutil.rmtree(dst)
                                shutil.copytree(src, dst)
                        print(f"  Removing temporary experiments/ folder")
                        shutil.rmtree(experiments_path)

                # Step 5: Restore params.yaml
                if backup_path.exists():
                    print(f"  Restoring params.backup.yaml -> params.yaml")
                    shutil.copy2(backup_path, params_path)

                print(f"  ✓ Completed")

            except subprocess.CalledProcessError as e:
                print(f"  ✗ Training failed with exit code {e.returncode}")
                # Restore params.yaml before failing fast
                if backup_path.exists():
                    print(f"  Restoring params.backup.yaml -> params.yaml")
                    shutil.copy2(backup_path, params_path)
                # Clean up experiments folder if it exists
                if experiments_path.exists():
                    print(f"  Cleaning up temporary experiments/ folder")
                    shutil.rmtree(experiments_path)
                print(f"\nFailing fast due to error in: {output_dir}")
                sys.exit(1)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                # Restore params.yaml before failing fast
                if backup_path.exists():
                    print(f"  Restoring params.backup.yaml -> params.yaml")
                    shutil.copy2(backup_path, params_path)
                # Clean up experiments folder if it exists
                if experiments_path.exists():
                    print(f"  Cleaning up temporary experiments/ folder")
                    shutil.rmtree(experiments_path)
                print(f"\nFailing fast due to error in: {output_dir}")
                sys.exit(1)

        print(f"\n✓ All {len(matching_items)} resume runs completed successfully!")

    else:
        # NORMAL MODE: Process config files
        for i, config_file in enumerate(matching_items, 1):
            config_path = Path(config_file)
            config_basename = config_path.stem
            output_folder = Path(config_basename)

            print(f"\n[{i}/{len(matching_items)}] Processing: {config_file}")

            try:
                # Step 1: Backup current params.yaml
                if params_path.exists():
                    print(f"  Backing up params.yaml -> params.backup.yaml")
                    shutil.copy2(params_path, backup_path)

                # Step 2: Copy config file to params.yaml
                print(f"  Copying config file to params.yaml")
                shutil.copy2(config_path, params_path)

                # Step 3: Run recursive training
                cmd = ["uv", "run", "recursive_train.py", "--generations", str(args.generations)]
                if args.force:
                    cmd.append("--force")
                if args.verbose:
                    cmd.append("--verbose")
                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True)

                # Step 4: Move experiments folder
                if experiments_path.exists():
                    if output_folder.exists():
                        print(f"  Removing existing {output_folder}/")
                        shutil.rmtree(output_folder)
                    print(f"  Moving experiments/ -> {output_folder}/")
                    shutil.move(str(experiments_path), str(output_folder))

                # Step 5: Delete source config file
                print(f"  Deleting source config file: {config_file}")
                config_path.unlink()

                # Step 6: Restore params.yaml
                if backup_path.exists():
                    print(f"  Restoring params.backup.yaml -> params.yaml")
                    shutil.copy2(backup_path, params_path)

                print(f"  ✓ Completed")

            except subprocess.CalledProcessError as e:
                print(f"  ✗ Training failed with exit code {e.returncode}")
                # Restore params.yaml before failing fast
                if backup_path.exists():
                    print(f"  Restoring params.backup.yaml -> params.yaml")
                    shutil.copy2(backup_path, params_path)
                print(f"\nFailing fast due to error in: {config_file}")
                sys.exit(1)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                # Restore params.yaml before failing fast
                if backup_path.exists():
                    print(f"  Restoring params.backup.yaml -> params.yaml")
                    shutil.copy2(backup_path, params_path)
                print(f"\nFailing fast due to error in: {config_file}")
                sys.exit(1)

        print(f"\n✓ All {len(matching_items)} training runs completed successfully!")


if __name__ == "__main__":
    main()
