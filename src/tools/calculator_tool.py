"""落地价计算工具：购置税（新能源减免）+ 保险 + 上牌费 + 贷款方案。

购置税口径（财政部 税务总局 工业和信息化部公告 2023 年第 10 号及后续延续公告）：
  2024-01-01 ~ 2025-12-31  免征，每辆减免不超过 3 万元
  2026-01-01 ~ 2027-12-31  减半征收，每辆减免不超过 1.5 万元
  其他时间                  按 10% 全额征收
计税价格 = 含税裸车价 / 1.13，购置税 = 计税价格 × 10%（再减免）。
"""
from __future__ import annotations

from datetime import date

from langchain_core.tools import tool

import config

EXEMPT_WINDOW = (date(2024, 1, 1), date(2025, 12, 31))
HALF_WINDOW = (date(2026, 1, 1), date(2027, 12, 31))
VAT_RATE = 1.13
TAX_RATE = 0.10


def purchase_tax(price_yuan: float, on: date | None = None) -> tuple[float, str]:
    """购置税金额与说明。"""
    today = on or date.today()
    taxable = price_yuan / VAT_RATE
    full_tax = taxable * TAX_RATE

    if EXEMPT_WINDOW[0] <= today <= EXEMPT_WINDOW[1]:
        tax = max(0.0, full_tax - 30000)
        note = "免征（减免上限 3 万元）"
    elif HALF_WINDOW[0] <= today <= HALF_WINDOW[1]:
        reduction = min(full_tax * 0.5, 15000)
        tax = full_tax - reduction
        note = "减半征收（减免上限 1.5 万元）"
    else:
        tax = full_tax
        note = "全额征收"
    return round(tax, 2), note


def monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """等额本息月供。"""
    rate = annual_rate / 12
    if rate == 0:
        return round(principal / months, 2)
    factor = (1 + rate) ** months
    return round(principal * rate * factor / (factor - 1), 2)


def to_yuan(price: float) -> float:
    """容错：裸车价 > 1000 视为「元」，否则视为「万元」。"""
    return price if price > 1000 else price * 10000


@tool
def calc_landing_price(
    price: float,
    down_payment_ratio: float | None = None,
    months: int | None = None,
    annual_rate: float | None = None,
) -> str:
    """计算新能源车落地价，含购置税、保险、上牌费与贷款月供。

    Args:
        price: 裸车指导价/成交价。传入 20 表示 20 万元，传入 200000 表示 20 万元。
        down_payment_ratio: 首付比例，例如 0.3 表示首付三成；留空用配置默认值。
        months: 贷款期数（月）；留空用配置默认值。
        annual_rate: 年化利率，例如 0.045；留空用配置默认值。
    """
    cfg = config.get_config()["calculator"]
    car_price = to_yuan(price)
    tax, tax_note = purchase_tax(car_price)
    commercial = max(cfg["commercial_insurance_min"], round(car_price * cfg["commercial_insurance_rate"], 2))
    insurance = cfg["compulsory_insurance"] + commercial
    plate = cfg["plate_fee"]
    total = car_price + tax + insurance + plate

    ratio = down_payment_ratio if down_payment_ratio is not None else cfg["loan"]["down_payment_ratio"]
    term = int(months if months is not None else cfg["loan"]["months"])
    rate = annual_rate if annual_rate is not None else cfg["loan"]["annual_rate"]
    down_payment = round(car_price * ratio, 2)
    principal = car_price - down_payment
    monthly = monthly_payment(principal, rate, term)
    total_loan_payment = round(monthly * term, 2)
    interest = round(total_loan_payment - principal, 2)
    first_payment = round(down_payment + tax + insurance + plate + monthly, 2)

    def money(value: float) -> str:
        return f"{value:,.0f} 元"

    return "\n".join(
        [
            f"裸车价：{money(car_price)}",
            "",
            "| 项目 | 金额 | 说明 |",
            "| --- | --- | --- |",
            f"| 购置税 | {money(tax)} | {tax_note}，计税价格按含税价 ÷ 1.13 |",
            f"| 交强险 | {money(cfg['compulsory_insurance'])} | 6 座以下家用车 |",
            f"| 商业险 | {money(commercial)} | 按裸车价 {cfg['commercial_insurance_rate']:.0%} 估算，保额与折扣以保险公司报价为准 |",
            f"| 上牌服务费 | {money(plate)} | 配置项 calculator.plate_fee |",
            f"| **全款落地价** | **{money(total)}** | 裸车价 + 购置税 + 保险 + 上牌费 |",
            "",
            f"贷款方案：首付 {ratio:.0%}（{money(down_payment)}），贷款 {money(principal)}，"
            f"{term} 期，年化 {rate:.2%}",
            f"- 等额本息月供：{money(monthly)}",
            f"- 利息总额：{money(interest)}（含利息总还款 {money(total_loan_payment)}）",
            f"- 提车首笔支出：{money(first_payment)}（首付 + 税费保险 + 首月月供）",
            "",
            "以上为估算口径，实际费用以当地政策、保险公司报价与经销商方案为准。",
        ]
    )

