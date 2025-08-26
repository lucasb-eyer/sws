import sws


def test_finalconfig_str_pretty_colors_and_formatting():
    c = sws.Config(lr=3, model={"width": 128, "depth": 4, "mup": 0.3})
    f = c.finalize()
    s = str(f)

    # ANSI helpers expected by the pretty-printer
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    BLUE = "\x1b[34m"
    RST = "\x1b[0m"

    # Expect sorted full keys with dimmed prefix, bold leaf, blue value
    expected_lines = [
        f"{BOLD}lr{RST}: {BLUE}3{RST}",
        f"{DIM}model.{RST}{BOLD}depth{RST}: {BLUE}4{RST}",
        f"{DIM}model.{RST}{BOLD}mup{RST}: {BLUE}0.3{RST}",
        f"{DIM}model.{RST}{BOLD}width{RST}: {BLUE}128{RST}",
    ]
    assert s.splitlines() == expected_lines

    # Empty config renders as {}
    assert str(sws.Config().finalize()) == "{}"
