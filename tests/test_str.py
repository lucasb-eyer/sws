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


def test_finalconfig_str_annotates_argv_overrides():
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    BLUE = "\x1b[34m"
    RST = "\x1b[0m"

    c = sws.Config(lr=3, model={"width": 128, "name": "x"})
    f = c.finalize(["lr=0.1", "name=vit_b16"])

    expected_lines = [
        f"{BOLD}lr{RST}: {BLUE}0.1{RST} {DIM}(argv){RST}",
        f"{DIM}model.{RST}{BOLD}name{RST}: {BLUE}'vit_b16'{RST} {DIM}(argv, as string){RST}",
        f"{DIM}model.{RST}{BOLD}width{RST}: {BLUE}128{RST}",
    ]
    assert str(f).splitlines() == expected_lines

    # Subviews annotate their own keys too.
    assert "(argv, as string)" in str(f.model)
    assert "(argv)" not in str(f.model).splitlines()[-1]  # width untouched

    # Children of an unpacked dict override are all marked.
    f2 = sws.Config(model={"width": 128}).finalize(["model=dict(width=64, depth=2)"])
    assert all("(argv)" in line for line in str(f2).splitlines())
