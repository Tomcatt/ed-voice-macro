#!/usr/bin/env python3
"""
ed-macro CLI — profile management

Usage:
  ed-macro profile list
  ed-macro profile load <name>
  ed-macro profile export <name> <dest>
  ed-macro profile import <src>
  ed-macro profile show [name]
"""
import sys
import shutil
from pathlib import Path
import yaml

CONFIG_DIR          = Path.home() / ".config" / "ed-voice-macro"
PROFILES_DIR        = CONFIG_DIR / "profiles"
ACTIVE_PROFILE_PATH = CONFIG_DIR / "active_profile"
PROFILE_EXT         = ".yaml"


def cmd_list(_args):
    active = ACTIVE_PROFILE_PATH.read_text().strip() if ACTIVE_PROFILE_PATH.exists() else ""
    profiles = sorted(PROFILES_DIR.glob(f"*{PROFILE_EXT}"))
    if not profiles:
        print("No profiles found.")
        return
    for p in profiles:
        name = p.stem
        marker = " *" if name == active else ""
        try:
            meta = yaml.safe_load(p.read_text())
            desc = meta.get("description", "")
            print(f"  {name}{marker}  —  {desc}")
        except Exception:
            print(f"  {name}{marker}")


def cmd_load(args):
    if not args:
        die("Usage: ed-macro profile load <name>")
    name = args[0]
    path = PROFILES_DIR / f"{name}{PROFILE_EXT}"
    if not path.exists():
        die(f"Profile not found: {name}")
    ACTIVE_PROFILE_PATH.write_text(name + "\n")
    print(f"Active profile set to: {name}")
    print("(daemon will hot-swap within 1 second if running)")


def cmd_export(args):
    if len(args) < 2:
        die("Usage: ed-macro profile export <name> <dest.yaml>")
    name, dest = args[0], Path(args[1])
    src = PROFILES_DIR / f"{name}{PROFILE_EXT}"
    if not src.exists():
        die(f"Profile not found: {name}")
    shutil.copy2(src, dest)
    print(f"Exported: {src} → {dest}")


def cmd_import(args):
    if not args:
        die("Usage: ed-macro profile import <file.yaml>")
    src = Path(args[0])
    if not src.exists():
        die(f"File not found: {src}")
    try:
        meta = yaml.safe_load(src.read_text())
        if "name" not in meta:
            die("Invalid profile: missing 'name' field")
    except Exception as e:
        die(f"Could not parse profile: {e}")
    dest = PROFILES_DIR / src.name
    if dest.exists():
        confirm = input(f"Profile '{dest.stem}' already exists. Overwrite? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return
    shutil.copy2(src, dest)
    print(f"Imported: {dest.stem}")


def cmd_show(args):
    if args:
        name = args[0]
    elif ACTIVE_PROFILE_PATH.exists():
        name = ACTIVE_PROFILE_PATH.read_text().strip()
    else:
        die("No active profile set")
    path = PROFILES_DIR / f"{name}{PROFILE_EXT}"
    if not path.exists():
        die(f"Profile not found: {name}")
    print(path.read_text())


def die(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


CMDS = {
    "list":   cmd_list,
    "load":   cmd_load,
    "export": cmd_export,
    "import": cmd_import,
    "show":   cmd_show,
}

def main():
    args = sys.argv[1:]
    if len(args) < 2 or args[0] != "profile" or args[1] not in CMDS:
        print(__doc__)
        sys.exit(0)
    CMDS[args[1]](args[2:])

if __name__ == "__main__":
    main()
