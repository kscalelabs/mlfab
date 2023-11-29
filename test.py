"""Curses test program."""

import time
from curses import wrapper
from typing import TYPE_CHECKING

import mlfab

if TYPE_CHECKING:
    from _curses import _CursesWindow


def main(stdscr: "_CursesWindow") -> None:
    while True:
        stdscr.clear()

        blocks = [
            [
                mlfab.TextBlock("Hello, world!"),
            ],
            [
                mlfab.TextBlock("This is a test."),
                mlfab.TextBlock("Another."),
            ],
            [
                mlfab.TextBlock("Current time:", color="light-green", bold=True),
                mlfab.TextBlock(time.strftime("%H:%M:%S")),
            ],
        ]
        blocks_str = mlfab.render_text_blocks(blocks)

        stdscr.addstr(blocks_str)

        stdscr.refresh()
        time.sleep(1.0)


wrapper(main)
