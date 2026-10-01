"""Automated Indian-language video dubbing."""


def main() -> int:
    from .cli import main as cli_main

    return cli_main()
