#!/usr/bin/env python3
"""
Batch training CLI for sdpype.

Enables running recursive_train.py across multiple configuration files:
1. Backup current params.yaml
2. Copy config file to params.yaml
3. Run recursive training
4. Move experiments folder
5. Delete source config file
6. Restore params.yaml
"""

import argparse
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Batch training CLI for sdpype",
        epilog="Example: uv run batch_train.py './configs/params/*rtvae*.yaml' --generations 20"
    )
    parser.add_argument(
        "pattern",
        help="Glob pattern for config files (e.g., './configs/params/*rtvae*.yaml')"
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

    args = parser.parse_args()

    # Get matching files
    matching_files = sorted(glob(args.pattern))

    if not matching_files:
        print(f"No files matching pattern: {args.pattern}")
        sys.exit(1)

    # Print dry-run info
    if args.dry_run:
        print(f"DRY-RUN MODE")
        print(f"Pattern: {args.pattern}")
        print(f"Generations: {args.generations}")
        print(f"Force: {args.force}")
        print(f"\nMatching files ({len(matching_files)}):")
        for i, filepath in enumerate(matching_files, 1):
            basename = Path(filepath).stem
            print(f"  {i}. {filepath}")
            print(f"     -> Output folder: {basename}/")

        force_flag = " --force" if args.force else ""
        print(f"\nWorkflow for each file:")
        print(f"  1. Backup params.yaml -> params.backup.yaml")
        print(f"  2. Copy config file -> params.yaml")
        print(f"  3. Run: uv run recursive_train.py --generations {args.generations}{force_flag}")
        print(f"  4. Move experiments/ -> {{config_basename}}/")
        print(f"  5. Delete source config file")
        print(f"  6. Restore params.backup.yaml -> params.yaml")
        print(f"  7. Continue to next file")
        return

    # Normal execution
    params_path = Path("params.yaml")
    backup_path = Path("params.backup.yaml")
    experiments_path = Path("experiments")

    for i, config_file in enumerate(matching_files, 1):
        config_path = Path(config_file)
        config_basename = config_path.stem
        output_folder = Path(config_basename)

        print(f"\n[{i}/{len(matching_files)}] Processing: {config_file}")

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

    print(f"\n✓ All {len(matching_files)} training runs completed successfully!")


if __name__ == "__main__":
    main()
