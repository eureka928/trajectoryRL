#!/usr/bin/env python3
"""TrajectoryRL Miner — CLI entry point.

Subcommands:
    build     Build pack.json from AGENTS.md file
    validate  Validate a pack.json locally
    submit    Push pack to GitHub + submit on-chain commitment
    status    Check current on-chain commitment

Examples:
    # Build a pack from your AGENTS.md
    python neurons/miner.py build --agents-md ./AGENTS.md --output pack.json

    # Validate before submitting
    python neurons/miner.py validate pack.json

    # Submit (pack already pushed to GitHub)
    python neurons/miner.py submit pack.json \\
        --repo myuser/my-pack \\
        --git-commit abc123def456... \\
        --wallet.name miner --wallet.hotkey default

    # Submit with auto-push (local git repo)
    python neurons/miner.py submit pack.json \\
        --repo myuser/my-pack \\
        --repo-path /path/to/local/repo \\
        --wallet.name miner --wallet.hotkey default

    # Check current commitment
    python neurons/miner.py status --wallet.name miner --wallet.hotkey default
"""

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


def cmd_build(args):
    """Build pack.json from AGENTS.md."""
    from trajectoryrl.base.miner import TrajectoryMiner

    pack = TrajectoryMiner.build_pack(
        agents_md=args.agents_md,
        pack_name=args.pack_name,
        pack_version=args.pack_version,
        soul_md=args.soul_md,
    )

    pack_hash = TrajectoryMiner.save_pack(pack, args.output)
    size = len(json.dumps(pack, sort_keys=True))

    print(f"Pack built: {args.output}")
    print(f"  Hash:  {pack_hash}")
    print(f"  Size:  {size} bytes (limit: 32768)")
    print(f"  Files: {list(pack['files'].keys())}")

    # Validate
    result = TrajectoryMiner.validate(pack)
    if result.passed:
        print("  Schema: PASSED")
    else:
        print("  Schema: FAILED")
        for issue in result.issues:
            print(f"    - {issue}")
        return 1

    return 0


def cmd_validate(args):
    """Validate a pack.json locally."""
    from trajectoryrl.base.miner import TrajectoryMiner

    pack = TrajectoryMiner.load_pack(args.pack_path)
    result = TrajectoryMiner.validate(pack)
    pack_hash = TrajectoryMiner.compute_pack_hash(pack)
    size = len(json.dumps(pack, sort_keys=True))

    print(f"Pack: {args.pack_path}")
    print(f"  Hash:    {pack_hash}")
    print(f"  Size:    {size} bytes (limit: 32768)")
    print(f"  Name:    {pack.get('metadata', {}).get('pack_name', '?')}")
    print(f"  Version: {pack.get('metadata', {}).get('pack_version', '?')}")
    print(f"  Files:   {list(pack.get('files', {}).keys())}")

    if result.passed:
        print("  Schema:  PASSED")
        return 0
    else:
        print("  Schema:  FAILED")
        for issue in result.issues:
            print(f"    - {issue}")
        return 1


def cmd_submit(args):
    """Submit pack to the network."""
    from trajectoryrl.base.miner import TrajectoryMiner

    miner = TrajectoryMiner(
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        netuid=args.netuid,
        network=args.network,
    )

    pack = TrajectoryMiner.load_pack(args.pack_path)

    success = miner.submit(
        pack=pack,
        repo=args.repo,
        git_commit=args.git_commit,
        repo_path=args.repo_path,
    )

    if success:
        pack_hash = TrajectoryMiner.compute_pack_hash(pack)
        print(f"Submitted successfully!")
        print(f"  Pack hash: {pack_hash}")
        print(f"  Repo:      {args.repo}")
        if args.git_commit:
            print(f"  Commit:    {args.git_commit}")
        return 0
    else:
        print("Submission failed. Check logs for details.")
        return 1


def cmd_status(args):
    """Check current on-chain commitment."""
    from trajectoryrl.base.miner import TrajectoryMiner
    from trajectoryrl.utils.commitments import parse_commitment

    miner = TrajectoryMiner(
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        netuid=args.netuid,
        network=args.network,
    )

    raw = miner.get_current_commitment()
    if raw is None:
        print("No commitment found on-chain.")
        return 1

    print(f"Raw commitment: {raw}")
    parsed = parse_commitment(raw)
    if parsed:
        pack_hash, git_commit, repo_url = parsed
        print(f"  Pack hash:  {pack_hash}")
        print(f"  Git commit: {git_commit}")
        print(f"  Repo:       {repo_url}")
    else:
        print("  (could not parse commitment)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="TrajectoryRL Miner — build and submit policy packs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # --- build ---
    p_build = sub.add_parser("build", help="Build pack.json from AGENTS.md")
    p_build.add_argument(
        "--agents-md", required=True,
        help="Path to AGENTS.md file",
    )
    p_build.add_argument(
        "--soul-md", default=None,
        help="Path to optional SOUL.md file",
    )
    p_build.add_argument(
        "--pack-name", default="my-pack",
        help="Pack name for metadata (default: my-pack)",
    )
    p_build.add_argument(
        "--pack-version", default="1.0.0",
        help="Semver version (default: 1.0.0)",
    )
    p_build.add_argument(
        "--output", "-o", default="pack.json",
        help="Output path (default: pack.json)",
    )

    # --- validate ---
    p_validate = sub.add_parser("validate", help="Validate pack.json locally")
    p_validate.add_argument("pack_path", help="Path to pack.json")

    # --- submit ---
    p_submit = sub.add_parser("submit", help="Submit pack on-chain")
    p_submit.add_argument("pack_path", help="Path to pack.json")
    p_submit.add_argument("--repo", required=True, help="GitHub repo (owner/repo)")
    p_submit.add_argument(
        "--git-commit", default=None,
        help="Git commit hash (if already pushed)",
    )
    p_submit.add_argument(
        "--repo-path", default=None,
        help="Local git repo path (for auto-push if --git-commit not given)",
    )
    _add_wallet_args(p_submit)

    # --- status ---
    p_status = sub.add_parser("status", help="Check on-chain commitment")
    _add_wallet_args(p_status)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "build": cmd_build,
        "validate": cmd_validate,
        "submit": cmd_submit,
        "status": cmd_status,
    }
    return handlers[args.command](args)


def _add_wallet_args(parser):
    """Add common Bittensor wallet/network args."""
    parser.add_argument("--wallet.name", dest="wallet_name", default="miner")
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    parser.add_argument("--netuid", type=int, default=11)
    parser.add_argument("--network", default="finney")


if __name__ == "__main__":
    sys.exit(main() or 0)
