"""Creates a new project, copying a standard format."""

import argparse
from pathlib import Path

import git


def get_default_author() -> str:
    try:
        return git.config.GitConfigParser().get_value("user", "name")  # type: ignore[return-value]
    except Exception:
        raise ValueError("Could not get configured Git author; please manually specify the --author argument")


def get_default_email() -> str:
    try:
        return git.config.GitConfigParser().get_value("user", "email")  # type: ignore[return-value]
    except Exception:
        raise ValueError("Could not get configured Git email; please manually specify the --email argument")


def cleanup_url(s: str) -> str:
    s = s.replace("git@", "https://")
    s = s.replace(".git", "")
    s = s.replace(":", "/")
    return s


def get_default_url() -> str:
    try:
        repo = git.Repo(search_parent_directories=True)
        return cleanup_url(repo.remote().url)
    except Exception:
        raise ValueError("Could not get configured Git URL; please manually specify the --url argument")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Project name")
    parser.add_argument("--description", help="Project description; if not provided, uses a generic default")
    parser.add_argument("--author", help="Project author; if not provided, defaults to the current user")
    parser.add_argument("--email", help="Project email; if not provided, defaults to <author>@users.noreply.github.com")
    parser.add_argument("--url", help="Project URL; if not provided, defaults to https://github.com/<author>/<name>")
    args = parser.parse_args()

    # Gets default values for the project metadata.
    name: str = args.name
    description: str = args.description or f"The {name} project"
    author: str = args.author or get_default_author()
    email: str = args.email or get_default_email()
    url: str = args.url or get_default_url()

    replacements = {
        "[[PROJECT NAME]]": name,
        "[[PROJECT DESCRIPTION]]": description,
        "[[PROJECT AUTHOR]]": author,
        "[[PROJECT EMAIL]]": email,
        "[[PROJECT URL]]": url,
    }

    # Copies the `assets` directory to the target directory, replacing placeholders.
    src_dir = (Path(__file__).parent / "assets").resolve()
    tgt_dir = (Path.cwd() / name).resolve()
    tgt_dir.mkdir(parents=True, exist_ok=True)

    def recursive_copy(src: Path, tgt: Path) -> None:
        if src.is_dir():
            if src.name in ("__pycache__", ".git", ".ruff_cache", ".pytest_cache"):
                return
            tgt.mkdir(parents=True, exist_ok=True)
            for src_file in src.iterdir():
                recursive_copy(src_file, tgt / src_file.name)
        else:
            with src.open("r", encoding="utf-8") as f:
                src_contents = f.read()
            tgt_contents = src_contents
            for src_str, tgt_str in replacements.items():
                tgt_contents = tgt_contents.replace(src_str, tgt_str)
            with tgt.open("w", encoding="utf-8") as f:
                f.write(tgt_contents)

    recursive_copy(src_dir, tgt_dir)

    # Renames the `project` subdirectory to the project name.
    (tgt_dir / "project").rename(tgt_dir / name)


if __name__ == "__main__":
    main()
