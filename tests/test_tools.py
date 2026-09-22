"""工具层测试：落地价计算（涉及金额，必须有校验）与联网搜索降级。"""
from __future__ import annotations

import re
import time
from datetime import date

import pytest

import config
from src.tools import search_tool
from src.tools.calculator_tool import calc_landing_price, monthly_payment, purchase_tax, to_yuan


def test_purchase_tax_rule_per_period():
    # 2024-2025：免征，减免上限 3 万元
    assert purchase_tax(200_000, date(2025, 6, 1))[0] == 0
    assert purchase_tax(600_000, date(2025, 6, 1))[0] == pytest.approx(
        600_000 / 1.13 * 0.10 - 30_000, abs=0.01
    )
    # 2026-2027：减半征收，减免上限 1.5 万元
    assert purchase_tax(200_000, date(2026, 6, 1))[0] == pytest.approx(
        200_000 / 1.13 * 0.05, abs=0.01
    )
    assert purchase_tax(400_000, date(2026, 6, 1))[0] == pytest.approx(
        400_000 / 1.13 * 0.10 - 15_000, abs=0.01
    )
    # 政策窗口之外：全额征收
    assert purchase_tax(200_000, date(2023, 6, 1))[0] == pytest.approx(
        200_000 / 1.13 * 0.10, abs=0.01
    )


def test_landing_price_matches_manual_calculation():
    text = calc_landing_price.invoke({"price": 20})
    cfg = config.get_config()["calculator"]
    car_price = 200_000
    tax = car_price / 1.13 * 0.05
    insurance = cfg["compulsory_insurance"] + max(
        cfg["commercial_insurance_min"], round(car_price * cfg["commercial_insurance_rate"], 2)
    )
    total = car_price + tax + insurance + cfg["plate_fee"]
    principal = car_price * (1 - cfg["loan"]["down_payment_ratio"])
    monthly = monthly_payment(principal, cfg["loan"]["annual_rate"], cfg["loan"]["months"])

    amounts = [int(value.replace(",", "")) for value in re.findall(r"([\d,]+) 元", text)]
    assert any(abs(amount - total) <= 1 for amount in amounts), text
    assert any(abs(amount - monthly) <= 1 for amount in amounts), text
    assert "全款落地价" in text and "等额本息月供" in text


def test_landing_price_accepts_wan_and_yuan():
    assert to_yuan(20) == 200_000          # 20 万元
    assert to_yuan(200_000) == 200_000     # 200000 元


def test_monthly_payment_zero_rate():
    assert monthly_payment(100_000, 0, 10) == 10_000.0


def test_web_search_degrades_when_unavailable(monkeypatch):
    def boom(query: str, max_results: int):
        raise RuntimeError("network down")

    monkeypatch.setattr(search_tool, "_search_blocking", boom)
    result = search_tool.web_search.invoke({"query": "小米 SU7 月销量"})
    assert "本地知识库" in result and "联网搜索失败" in result


def test_web_search_times_out_fast(monkeypatch):
    monkeypatch.setitem(config.get_config()["search"], "timeout", 0.05)

    def slow(query: str, max_results: int):
        time.sleep(1)
        return []

    monkeypatch.setattr(search_tool, "_search_blocking", slow)
    started = time.monotonic()
    result = search_tool.web_search.invoke({"query": "小米 SU7 月销量"})
    assert "未返回结果" in result and "本地知识库" in result
    assert time.monotonic() - started < 0.5, "超时后不应继续阻塞等待结果"
