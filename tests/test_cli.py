from etf_momentum_backtest.cli import main


def test_cli_uses_default_tickers(
    capsys,
) -> None:
    main([])

    output = capsys.readouterr().out

    assert "SPY, QQQ, TLT, IEF, GLD" in output
    assert "Top K: 2" in output


def test_cli_accepts_custom_tickers(
    capsys,
) -> None:
    main(
        [
            "--tickers",
            "spy",
            "iwm",
            "gld",
            "--top-k",
            "2",
            "--cost-bps",
            "5",
        ]
    )

    output = capsys.readouterr().out

    assert "SPY, IWM, GLD" in output
    assert "Top K: 2" in output
    assert "Transaction cost: 0.05%" in output
