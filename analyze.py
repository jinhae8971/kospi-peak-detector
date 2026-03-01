"""
analyze.py — 코스피 중장기 고점 판독기 v2.1
=============================================
4대 핵심 프레임워크 기반 전문 수준 자동 분석:
  1. 밸류에이션 & 과열 지표 (Valuation & Overheat)
     - 버핏 지수 (GDP 대비 시가총액 비율)
     - 12개월 선행 PER / PBR (yfinance 기반)
     - 12개월 이동평균 이격도
  2. 반도체 업황 & 수출 데이터 (Semiconductor Cycle)
     - 삼성전자 / SK하이닉스 주가 추세 및 EPS 방향성
     - SOXX / NVDA 글로벌 반도체 사이클
     - HBM/AI 인프라 수요 프록시
  3. 수급 & 심리 지표 (Supply & Demand)
     - 코스피 실현 변동성 (VKOSPI 프록시)
     - 원/달러 환율 (외국인 자금 이탈 프록시)
     - EWY (외국인 수급 프록시) 추세
     - KODEX 레버리지 ETF 거래량 (신용융자 잔고 프록시)
  4. 기술적 분석 (Technical Analysis)
     - RSI(14), MACD, 볼린저밴드 %B
     - 이동평균선 배열 (5/20/60/120/200일)
     - 엘리어트 파동 위치 추정
     - 12개월 이격도 (핵심 고점 판별 지표)

출력: reports/latest.json (대시보드 연동)
"""

import os
import json
import time
import math
import requests
import yfinance as yf
from datetime import datetime, timezone, timedelta
from openai import OpenAI

client = OpenAI()
KST = timezone(timedelta(hours=9))

# ─────────────────────────────────────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────────────────────────────────────

def get_history(symbol, period='2y'):
    """yfinance로 일별 종가 배열 반환."""
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period)
        if hist.empty:
            print(f'  [데이터 없음] {symbol}')
            return None
        closes  = hist['Close'].dropna().tolist()
        volumes = hist['Volume'].dropna().tolist()
        return {
            'symbol':  symbol,
            'price':   closes[-1] if closes else None,
            'closes':  closes,
            'volumes': volumes,
        }
    except Exception as e:
        print(f'  [yfinance 오류] {symbol}: {e}')
        return None


def get_info(symbol):
    """yfinance로 종목 기본 정보 반환."""
    try:
        t = yf.Ticker(symbol)
        return t.info or {}
    except Exception as e:
        print(f'  [info 오류] {symbol}: {e}')
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 기술 지표 계산 함수
# ─────────────────────────────────────────────────────────────────────────────

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def calc_ema(closes, period):
    if len(closes) < period:
        return None
    k = 2 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1 - k)
    return ema


def calc_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None
    macd_line = ema_fast - ema_slow
    # 시그널: 최근 구간의 MACD 값들로 EMA 근사
    macd_series = []
    n = len(closes)
    for i in range(signal + slow, 0, -1):
        if i > n:
            continue
        sub = closes[:n - i + 1]
        ef = calc_ema(sub, fast)
        es = calc_ema(sub, slow)
        if ef and es:
            macd_series.append(ef - es)
    signal_line = calc_ema(macd_series, signal) if len(macd_series) >= signal else None
    histogram = macd_line - signal_line if signal_line else None
    return (round(macd_line, 2),
            round(signal_line, 2) if signal_line else None,
            round(histogram, 2) if histogram else None)


def calc_bollinger(closes, period=20, std_mult=2):
    if len(closes) < period:
        return None, None, None, None
    window = closes[-period:]
    ma = sum(window) / period
    std = math.sqrt(sum((c - ma) ** 2 for c in window) / period)
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    price = closes[-1]
    pct_b = (price - lower) / (upper - lower) * 100 if (upper - lower) > 0 else None
    return round(upper, 2), round(ma, 2), round(lower, 2), round(pct_b, 2) if pct_b else None


def calc_disparity(closes, period):
    """이격도: 현재가 / N일 이동평균 × 100."""
    if len(closes) < period:
        return None, None
    ma = sum(closes[-period:]) / period
    disp = closes[-1] / ma * 100
    return round(ma, 2), round(disp, 2)


def calc_realized_vol(closes, period=20):
    """실현 변동성 (연환산 %) — VKOSPI 프록시."""
    if len(closes) < period + 1:
        return None
    log_returns = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
    rv = math.sqrt(sum(r**2 for r in log_returns[-period:]) / period) * math.sqrt(252) * 100
    return round(rv, 2)


def estimate_elliott_wave(closes):
    """엘리어트 파동 위치 추정 (2년 고점/저점 패턴 기반)."""
    if len(closes) < 100:
        return {'wave_position': '데이터 부족', 'wave_risk': 'unknown', 'wave_desc': ''}

    high_2y = max(closes)
    low_2y  = min(closes)
    current = closes[-1]

    pct_from_2y_high = (current / high_2y - 1) * 100
    pct_from_2y_low  = (current / low_2y - 1) * 100

    high_1y = max(closes[-252:]) if len(closes) >= 252 else max(closes)
    pct_from_1y_high = (current / high_1y - 1) * 100

    if pct_from_2y_high > -3 and pct_from_2y_low > 60:
        wave_position = '5파 말미 / 고점권 강력 경고'
        wave_risk = 'DANGER'
        wave_desc = (f'2년 저점 대비 +{pct_from_2y_low:.1f}% 상승, 2년 고점 대비 {pct_from_2y_high:.1f}% — '
                     f'상승 5파 말미 또는 연장파 끝자락 가능성. 과거 유사 사례(2007, 2011, 2021년) 대비 '
                     f'2~4개월 내 중장기 고점 형성 경고.')
    elif pct_from_2y_high > -8 and pct_from_2y_low > 40:
        wave_position = '5파 진행 중 / 고점 경계'
        wave_risk = 'CAUTION'
        wave_desc = (f'2년 저점 대비 +{pct_from_2y_low:.1f}% 상승, 2년 고점 대비 {pct_from_2y_high:.1f}% — '
                     f'상승 5파 진행 중. 고점 형성 경계 구간이나 추가 상승 여력 잔존.')
    elif pct_from_2y_high < -20:
        wave_position = 'A~C파 조정 진행 중'
        wave_risk = 'NORMAL'
        wave_desc = (f'2년 고점 대비 {pct_from_2y_high:.1f}% 조정 — '
                     f'조정 파동(A~C파) 진행 중. 중장기 매수 기회 탐색 구간.')
    elif pct_from_2y_high < -8:
        wave_position = '조정 초기 (4파 또는 A파)'
        wave_risk = 'WATCH'
        wave_desc = (f'2년 고점 대비 {pct_from_2y_high:.1f}% — '
                     f'단기 조정(4파) 또는 고점 이후 A파 하락 초기 가능성.')
    else:
        wave_position = '3~4파 구간 / 추세 지속'
        wave_risk = 'WATCH'
        wave_desc = (f'2년 고점 대비 {pct_from_2y_high:.1f}% — '
                     f'중기 상승 추세 유지. 3파 또는 4파 조정 후 재상승 구간.')

    return {
        'wave_position':    wave_position,
        'wave_risk':        wave_risk,
        'wave_desc':        wave_desc,
        'pct_from_2y_high': round(pct_from_2y_high, 2),
        'pct_from_1y_high': round(pct_from_1y_high, 2),
        'pct_from_2y_low':  round(pct_from_2y_low, 2),
        'high_2y':          round(high_2y, 2),
        'low_2y':           round(low_2y, 2),
        'high_1y':          round(high_1y, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 프레임워크 1: 밸류에이션 & 과열 지표
# ─────────────────────────────────────────────────────────────────────────────

def analyze_valuation():
    print('\n━━━ [1] 밸류에이션 & 과열 지표 ━━━')
    result = {}
    warnings = []

    # 코스피 지수 (2년 데이터)
    kospi = get_history('^KS11', period='2y')
    if kospi:
        closes  = kospi['closes']
        current = kospi['price']
        result['kospi_price'] = round(current, 2) if current else None
        print(f'  코스피 현재가: {current:,.2f}pt ({len(closes)}일 데이터)')

        # 이격도: 12개월(252일), 200일, 60일
        for period, label in [(252, '12개월'), (200, '200일'), (60, '3개월')]:
            ma, disp = calc_disparity(closes, period)
            if ma and disp:
                result[f'ma_{period}d'] = ma
                result[f'disparity_{period}d'] = disp
                print(f'  {label} 이격도: {disp:.1f}% (MA: {ma:,.2f})')

        # 52주 고저
        w52 = closes[-252:] if len(closes) >= 252 else closes
        result['kospi_52w_high'] = round(max(w52), 2)
        result['kospi_52w_low']  = round(min(w52), 2)
        pct_from_high = (current / max(w52) - 1) * 100
        result['pct_from_52w_high'] = round(pct_from_high, 2)
        print(f'  52주 고점: {max(w52):,.2f} / 현재 대비: {pct_from_high:.1f}%')

    # 삼성전자 PER/PBR/시총 (yfinance)
    samsung_info = get_info('005930.KS')
    if samsung_info:
        samsung_mcap = samsung_info.get('marketCap', 0) / 1e12  # 조 원
        result['samsung_market_cap_trillion'] = round(samsung_mcap, 1)
        result['samsung_forward_per']    = samsung_info.get('forwardPE')
        result['samsung_trailing_per']   = samsung_info.get('trailingPE')
        result['samsung_pbr']            = samsung_info.get('priceToBook')
        result['samsung_earnings_growth']= samsung_info.get('earningsGrowth')
        result['samsung_revenue_growth'] = samsung_info.get('revenueGrowth')
        result['samsung_target_price']   = samsung_info.get('targetMeanPrice')
        result['samsung_rec_mean']       = samsung_info.get('recommendationMean')
        print(f'  삼성전자 시총: {samsung_mcap:.1f}조 / Forward PER: {samsung_info.get("forwardPE")} / PBR: {samsung_info.get("priceToBook")}')

        if samsung_info.get('forwardPE') and samsung_info['forwardPE'] > 15:
            warnings.append(f'⚠️ 삼성전자 Forward PER {samsung_info["forwardPE"]:.1f}배 — 역사적 상단 초과 (기준: 13배)')
        if samsung_info.get('earningsGrowth') is not None and samsung_info['earningsGrowth'] < 0:
            warnings.append(f'⚠️ 삼성전자 이익 성장률 {samsung_info["earningsGrowth"]*100:.1f}% — 이익 컨센서스 하향 신호')

    # SK하이닉스 PER/PBR/시총
    hynix_info = get_info('000660.KS')
    if hynix_info:
        hynix_mcap = hynix_info.get('marketCap', 0) / 1e12
        result['hynix_market_cap_trillion'] = round(hynix_mcap, 1)
        result['hynix_forward_per']    = hynix_info.get('forwardPE')
        result['hynix_trailing_per']   = hynix_info.get('trailingPE')
        result['hynix_pbr']            = hynix_info.get('priceToBook')
        result['hynix_earnings_growth']= hynix_info.get('earningsGrowth')
        result['hynix_target_price']   = hynix_info.get('targetMeanPrice')
        print(f'  SK하이닉스 시총: {hynix_mcap:.1f}조 / Forward PER: {hynix_info.get("forwardPE")} / PBR: {hynix_info.get("priceToBook")}')

    # 버핏 지수 계산
    # 코스피 전체 시총 추정:
    # 삼성전자 + SK하이닉스 합산 시총 / 두 종목의 코스피 내 비중으로 역산
    # 2026년 초 AI 랠리로 두 종목 합산 비중이 약 60~65% 수준으로 추정
    KOREA_GDP_TRILLION_KRW = 2236  # 2024년 한국 명목 GDP (조 원)
    samsung_mcap_val = samsung_info.get('marketCap', 0) / 1e12 if samsung_info else 0
    hynix_mcap_val   = hynix_info.get('marketCap', 0) / 1e12 if hynix_info else 0
    combined_mcap    = samsung_mcap_val + hynix_mcap_val

    if combined_mcap > 0:
        # 두 종목 합산 비중 추정: 코스피 지수 수준에 따라 동적 조정
        # 코스피 6,000pt+ 수준에서 반도체 쏠림 심화 → 비중 약 60~65%
        kospi_pt = result.get('kospi_price', 3000)
        if kospi_pt > 5000:
            weight_est = 0.62  # 고점 구간: 반도체 쏠림 심화
        elif kospi_pt > 3500:
            weight_est = 0.40  # 중간 구간
        else:
            weight_est = 0.35  # 저점 구간: 분산된 시총

        estimated_total_mcap = combined_mcap / weight_est
        buffett_ratio = estimated_total_mcap / KOREA_GDP_TRILLION_KRW * 100
        result['estimated_total_mcap_trillion'] = round(estimated_total_mcap, 0)
        result['buffett_ratio'] = round(buffett_ratio, 1)
        result['korea_gdp_trillion'] = KOREA_GDP_TRILLION_KRW
        result['samsung_hynix_combined_mcap'] = round(combined_mcap, 1)
        result['samsung_hynix_weight_est'] = weight_est
        print(f'  버핏 지수 추정: {buffett_ratio:.1f}% (삼성+하이닉스 {combined_mcap:.0f}조 / 비중 {weight_est*100:.0f}% 가정 → 전체 {estimated_total_mcap:.0f}조 / GDP {KOREA_GDP_TRILLION_KRW}조)')

        if buffett_ratio > 180:
            warnings.append(f'🔴 버핏 지수 {buffett_ratio:.1f}% — 역사적 과열 구간 (기준: 180%+, 일부 분석 기관 경고 수준)')
        elif buffett_ratio > 150:
            warnings.append(f'⚠️ 버핏 지수 {buffett_ratio:.1f}% — 주의 구간 (기준: 150%+)')
        elif buffett_ratio > 120:
            warnings.append(f'⚡ 버핏 지수 {buffett_ratio:.1f}% — 경계 구간 (기준: 120%+)')

    result['warnings']      = warnings
    result['warning_count'] = len(warnings)
    print(f'  → 경고 {len(warnings)}건')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 프레임워크 2: 반도체 업황 & 수출 데이터
# ─────────────────────────────────────────────────────────────────────────────

def analyze_semiconductor():
    print('\n━━━ [2] 반도체 업황 & 수출 데이터 ━━━')
    result = {}
    signals = []

    def stock_analysis(symbol, name):
        data = get_history(symbol, period='2y')
        if not data:
            return {}
        closes = data['closes']
        price  = data['price']
        res = {'price': round(price, 0) if price else None}

        for period, label in [(20, '20d'), (60, '60d'), (120, '120d'), (200, '200d')]:
            if len(closes) >= period:
                ma = sum(closes[-period:]) / period
                res[f'ma_{label}'] = round(ma, 0)
                res[f'above_ma_{label}'] = price > ma if price else None

        for period, label in [(20, '1m'), (60, '3m'), (120, '6m'), (252, '12m')]:
            if len(closes) >= period:
                ret = (closes[-1] / closes[-period] - 1) * 100
                res[f'ret_{label}'] = round(ret, 2)

        w52 = closes[-252:] if len(closes) >= 252 else closes
        res['pct_from_52w_high'] = round((price / max(w52) - 1) * 100, 2)
        res['52w_high'] = round(max(w52), 0)
        res['52w_low']  = round(min(w52), 0)

        print(f'  {name}: {price:,.0f} / 3개월 {res.get("ret_3m", "N/A")}% / 52주 고점 대비 {res.get("pct_from_52w_high", "N/A")}%')
        return res

    # 삼성전자
    sam = stock_analysis('005930.KS', '삼성전자')
    result['samsung'] = sam
    if sam.get('above_ma_60d') is False:
        signals.append('⚠️ 삼성전자 60일 이평선 하향 이탈 — 업황 둔화 선행 신호')
    if sam.get('above_ma_200d') is False:
        signals.append('🔴 삼성전자 200일 이평선 하향 이탈 — 중장기 하락 전환 신호')
    if sam.get('ret_3m') is not None and sam['ret_3m'] < -15:
        signals.append(f'🔴 삼성전자 3개월 수익률 {sam["ret_3m"]:.1f}% — 급격한 하락, EPS 하향 우려')
    elif sam.get('ret_3m') is not None and sam['ret_3m'] < -8:
        signals.append(f'⚠️ 삼성전자 3개월 수익률 {sam["ret_3m"]:.1f}% — 업황 둔화 가능성')

    # SK하이닉스
    hyx = stock_analysis('000660.KS', 'SK하이닉스')
    result['hynix'] = hyx
    if hyx.get('above_ma_60d') is False:
        signals.append('⚠️ SK하이닉스 60일 이평선 하향 이탈 — HBM 수요 둔화 우려')
    if hyx.get('ret_3m') is not None and hyx['ret_3m'] < -15:
        signals.append(f'🔴 SK하이닉스 3개월 수익률 {hyx["ret_3m"]:.1f}% — AI 인프라 투자 사이클 둔화 신호')

    # SOXX (필라델피아 반도체 ETF)
    soxx = get_history('SOXX', period='2y')
    if soxx:
        closes = soxx['closes']
        price  = soxx['price']
        result['soxx_price'] = round(price, 2) if price else None
        if len(closes) >= 60:
            result['soxx_ret_3m'] = round((closes[-1] / closes[-60] - 1) * 100, 2)
        if len(closes) >= 200:
            ma200 = sum(closes[-200:]) / 200
            result['soxx_above_ma200'] = price > ma200 if price else None
            result['soxx_ma200'] = round(ma200, 2)
            result['soxx_disparity_200d'] = round(price / ma200 * 100, 2)
        print(f'  SOXX: ${price:.2f} / 3개월 {result.get("soxx_ret_3m", "N/A")}% / 200일 MA 위: {result.get("soxx_above_ma200", "N/A")}')
        if result.get('soxx_above_ma200') is False:
            signals.append('🔴 SOXX 200일 이평선 하향 이탈 — 글로벌 반도체 사이클 하강 국면')

    # NVIDIA (HBM/AI 인프라 수요 직접 지표)
    nvda = get_history('NVDA', period='2y')
    if nvda:
        closes = nvda['closes']
        price  = nvda['price']
        result['nvda_price'] = round(price, 2) if price else None
        if len(closes) >= 60:
            result['nvda_ret_3m'] = round((closes[-1] / closes[-60] - 1) * 100, 2)
        if len(closes) >= 200:
            ma200 = sum(closes[-200:]) / 200
            result['nvda_above_ma200'] = price > ma200 if price else None
            result['nvda_disparity_200d'] = round(price / ma200 * 100, 2)
        print(f'  NVDA: ${price:.2f} / 3개월 {result.get("nvda_ret_3m", "N/A")}% / 200일 MA 위: {result.get("nvda_above_ma200", "N/A")}')
        if result.get('nvda_above_ma200') is False:
            signals.append('⚠️ NVIDIA 200일 이평선 하향 이탈 — AI 인프라 투자 수요 둔화 우려')
        if result.get('nvda_ret_3m') is not None and result['nvda_ret_3m'] < -20:
            signals.append(f'🔴 NVIDIA 3개월 수익률 {result["nvda_ret_3m"]:.1f}% — AI ROI 논란 현실화 가능성')

    # 하이닉스 vs 삼성 상대 강도 (HBM 프리미엄 쏠림 체크)
    if sam.get('ret_3m') is not None and hyx.get('ret_3m') is not None:
        rel = hyx['ret_3m'] - sam['ret_3m']
        result['hynix_vs_samsung_rel_3m'] = round(rel, 2)
        if rel > 20:
            signals.append(f'⚡ SK하이닉스 vs 삼성전자 상대강도 +{rel:.1f}%p — HBM 프리미엄 과도 집중, 쏠림 주의')
        print(f'  하이닉스-삼성 상대강도(3M): {rel:+.1f}%p')

    result['signals']      = signals
    result['signal_count'] = len(signals)
    print(f'  → 신호 {len(signals)}건')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 프레임워크 3: 수급 & 심리 지표
# ─────────────────────────────────────────────────────────────────────────────

def analyze_supply_demand():
    print('\n━━━ [3] 수급 & 심리 지표 ━━━')
    result = {}
    warnings = []

    # 코스피 실현 변동성 (VKOSPI 프록시)
    # Yahoo Finance에서 ^VKOSPI 조회 불가 → 코스피 일별 수익률 기반 실현 변동성으로 대체
    kospi = get_history('^KS11', period='1y')
    if kospi:
        closes = kospi['closes']
        rv_20  = calc_realized_vol(closes, 20)
        rv_60  = calc_realized_vol(closes, 60)
        if len(closes) > 1:
            all_returns = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
            rv_1y = math.sqrt(sum(r**2 for r in all_returns) / len(all_returns)) * math.sqrt(252) * 100
            result['realized_vol_1y'] = round(rv_1y, 2)
        result['realized_vol_20d'] = rv_20
        result['realized_vol_60d'] = rv_60
        print(f'  코스피 실현변동성: 20일={rv_20}% / 60일={rv_60}% (연환산, VKOSPI 프록시)')

        # 변동성 경고 판단
        if rv_20:
            if rv_20 > 40:
                warnings.append(f'🔴 코스피 20일 실현변동성 {rv_20:.1f}% — 극단적 공포 구간 (VKOSPI 50+ 수준)')
            elif rv_20 > 25:
                warnings.append(f'⚠️ 코스피 20일 실현변동성 {rv_20:.1f}% — 불안 구간 (VKOSPI 25~35 수준)')
            # 변동성 급등 (20일 >> 60일) = 공포 급증 신호
            if rv_60 and rv_20 > rv_60 * 1.5:
                warnings.append(f'⚠️ 단기 변동성 급등 ({rv_20:.1f}% vs 60일 {rv_60:.1f}%) — 시장 불안 심화, 하방 헤지 수요 증가')

    # 원/달러 환율 (외국인 자금 이탈 프록시)
    usdkrw = get_history('USDKRW=X', period='1y')
    if usdkrw:
        closes = usdkrw['closes']
        price  = usdkrw['price']
        result['usdkrw'] = round(price, 2) if price else None
        if len(closes) >= 60:
            ma60 = sum(closes[-60:]) / 60
            result['usdkrw_ma60'] = round(ma60, 2)
            result['usdkrw_above_ma60'] = price > ma60 if price else None
            result['usdkrw_ret_3m'] = round((closes[-1] / closes[-60] - 1) * 100, 2)
        if len(closes) >= 20:
            ma20 = sum(closes[-20:]) / 20
            result['usdkrw_ma20'] = round(ma20, 2)
        print(f'  USD/KRW: {price:,.2f} / 3개월 변화: {result.get("usdkrw_ret_3m", "N/A")}%')

        if result.get('usdkrw_above_ma60') and result.get('usdkrw_ret_3m', 0) > 5:
            warnings.append(f'🔴 원/달러 환율 {price:,.0f}원 (+{result["usdkrw_ret_3m"]:.1f}%) — 외국인 자금 이탈 가속 우려')
        elif result.get('usdkrw_above_ma60') and result.get('usdkrw_ret_3m', 0) > 2:
            warnings.append(f'⚠️ 원/달러 환율 상승 (+{result.get("usdkrw_ret_3m", 0):.1f}%) — 외국인 수급 악화 신호')

    # EWY (iShares MSCI South Korea ETF) — 외국인 수급 프록시
    ewy = get_history('EWY', period='2y')
    if ewy:
        closes = ewy['closes']
        price  = ewy['price']
        result['ewy_price'] = round(price, 2) if price else None
        for period, label in [(20, '1m'), (60, '3m'), (252, '12m')]:
            if len(closes) >= period:
                result[f'ewy_ret_{label}'] = round((closes[-1] / closes[-period] - 1) * 100, 2)
        if len(closes) >= 200:
            ma200 = sum(closes[-200:]) / 200
            result['ewy_above_ma200'] = price > ma200 if price else None
            result['ewy_ma200'] = round(ma200, 2)
            result['ewy_disparity_200d'] = round(price / ma200 * 100, 2)
        if len(closes) >= 20:
            ma20 = sum(closes[-20:]) / 20
            result['ewy_above_ma20'] = price > ma20 if price else None
        print(f'  EWY(외국인 수급): ${price:.2f} / 3개월 {result.get("ewy_ret_3m", "N/A")}% / 200일 MA 위: {result.get("ewy_above_ma200", "N/A")}')

        if result.get('ewy_above_ma200') is False:
            warnings.append('🔴 EWY 200일 이평선 하향 이탈 — 외국인 중장기 매도 전환 신호')
        elif result.get('ewy_above_ma20') is False:
            warnings.append('⚠️ EWY 20일 이평선 하향 이탈 — 외국인 단기 매도 증가')

    # 신용융자 잔고 프록시: KODEX 레버리지 ETF (122630.KS) 거래량 추세
    lev = get_history('122630.KS', period='1y')
    if lev:
        closes  = lev['closes']
        volumes = lev['volumes']
        price   = lev['price']
        result['kodex_lev_price'] = round(price, 0) if price else None
        if len(volumes) >= 60:
            vol_20d = sum(volumes[-20:]) / 20
            vol_60d = sum(volumes[-60:]) / 60
            vol_ratio = vol_20d / vol_60d if vol_60d > 0 else None
            result['kodex_lev_vol_ratio'] = round(vol_ratio, 2) if vol_ratio else None
            print(f'  KODEX 레버리지 거래량 비율(20일/60일): {vol_ratio:.2f}x' if vol_ratio else '  KODEX 레버리지: 데이터 부족')
            if vol_ratio and vol_ratio > 1.5:
                warnings.append(f'⚠️ 레버리지 ETF 거래량 급증 ({vol_ratio:.1f}x) — 개인 빚투 과열, 신용융자 잔고 급증 추정')
            elif vol_ratio and vol_ratio > 1.2:
                warnings.append(f'⚡ 레버리지 ETF 거래량 증가 ({vol_ratio:.1f}x) — 개인 투자자 과열 조짐')

    # VIX (미국 공포지수) — 글로벌 리스크 온/오프 신호
    vix = get_history('^VIX', period='1y')
    if vix:
        price  = vix['price']
        closes = vix['closes']
        result['vix'] = round(price, 2) if price else None
        if closes:
            result['vix_1m_avg'] = round(sum(closes[-20:]) / min(len(closes), 20), 2)
        print(f'  VIX(미국 공포지수): {price:.2f} (1개월 평균: {result.get("vix_1m_avg", "N/A")})')
        if price and price > 30:
            warnings.append(f'🔴 VIX {price:.1f} — 글로벌 공포 구간 (기준: 30+), 외국인 리스크오프 전환 우려')
        elif price and price > 20:
            warnings.append(f'⚠️ VIX {price:.1f} — 글로벌 불안 구간 (기준: 20~30)')

    result['warnings']      = warnings
    result['warning_count'] = len(warnings)
    print(f'  → 경고 {len(warnings)}건')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 프레임워크 4: 기술적 분석
# ─────────────────────────────────────────────────────────────────────────────

def analyze_technical():
    print('\n━━━ [4] 기술적 분석 ━━━')
    result = {}
    warnings = []

    kospi = get_history('^KS11', period='2y')
    if not kospi or not kospi['closes']:
        print('  코스피 데이터 없음')
        return result

    closes = kospi['closes']
    price  = kospi['price']
    result['kospi_price'] = round(price, 2) if price else None

    # RSI (14일)
    rsi = calc_rsi(closes, 14)
    result['rsi_14'] = rsi
    if rsi:
        print(f'  RSI(14): {rsi:.1f}')
        if rsi > 80:
            warnings.append(f'🔴 RSI {rsi:.1f} — 극단적 과매수 (기준: 80+, 급격한 되돌림 위험)')
        elif rsi > 70:
            warnings.append(f'⚠️ RSI {rsi:.1f} — 과매수 구간 (기준: 70+)')
        elif rsi < 30:
            warnings.append(f'🟢 RSI {rsi:.1f} — 과매도 구간 (반등 기회 탐색)')

    # 볼린저 밴드 (20일)
    bb_upper, bb_ma, bb_lower, bb_pct = calc_bollinger(closes, 20)
    result['bb_upper'] = bb_upper
    result['bb_lower'] = bb_lower
    result['bb_pct']   = bb_pct
    result['ma_20d']   = bb_ma
    if bb_pct is not None:
        print(f'  볼린저밴드 %B: {bb_pct:.1f}% (상단: {bb_upper:,.0f} / 하단: {bb_lower:,.0f})')
        if bb_pct > 95:
            warnings.append(f'🔴 볼린저밴드 %B {bb_pct:.1f}% — 극단적 과열, 밴드 상단 완전 돌파')
        elif bb_pct > 80:
            warnings.append(f'⚠️ 볼린저밴드 %B {bb_pct:.1f}% — 상단 밴드 근접 과열')

    # MACD (12, 26, 9)
    macd_line, signal_line, histogram = calc_macd(closes)
    result['macd']           = macd_line
    result['macd_signal']    = signal_line
    result['macd_histogram'] = histogram
    result['macd_bullish']   = macd_line > 0 if macd_line is not None else None
    if macd_line is not None:
        print(f'  MACD: {macd_line:.2f} / Signal: {signal_line} / Histogram: {histogram}')
        if histogram is not None and histogram < 0 and macd_line > 0:
            warnings.append(f'⚠️ MACD 히스토그램 음전환 ({histogram:.2f}) — 상승 모멘텀 약화 신호')
        elif macd_line is not None and macd_line < 0:
            warnings.append(f'⚠️ MACD 음전환 ({macd_line:.2f}) — 중기 하락 모멘텀')

    # 이동평균선 배열 (5/20/60/120/200일)
    ma_values = {}
    for period in [5, 20, 60, 120, 200]:
        if len(closes) >= period:
            ma = sum(closes[-period:]) / period
            ma_values[period] = round(ma, 2)
            result[f'ma_{period}d'] = round(ma, 2)

    if len(ma_values) >= 4:
        periods = sorted(ma_values.keys())
        perfect_align = all(ma_values[periods[i]] > ma_values[periods[i+1]]
                            for i in range(len(periods)-1))
        result['ma_perfect_align'] = perfect_align
        print(f'  이평선 정배열: {"✅ 정배열" if perfect_align else "❌ 비정배열"}')
        if ma_values:
            ma_str = ' / '.join(f'MA{p}: {ma_values[p]:,.0f}' for p in sorted(ma_values.keys()))
            print(f'  {ma_str}')

        # 골든/데드 크로스 (20일 vs 60일)
        if 20 in ma_values and 60 in ma_values and len(closes) >= 62:
            ma20_prev = sum(closes[-21:-1]) / 20
            ma60_prev = sum(closes[-61:-1]) / 60
            if ma_values[20] > ma_values[60] and ma20_prev <= ma60_prev:
                result['cross_signal'] = '골든크로스 (20일 > 60일)'
            elif ma_values[20] < ma_values[60] and ma20_prev >= ma60_prev:
                result['cross_signal'] = '데드크로스 (20일 < 60일)'
                warnings.append('🔴 데드크로스 발생 (20일 < 60일) — 중기 하락 전환 신호')

    # 12개월 이격도 (핵심 고점 판별 지표)
    ma_252, disp_252 = calc_disparity(closes, 252)
    if ma_252 and disp_252:
        result['ma_252d'] = ma_252
        result['disparity_252d'] = disp_252
        print(f'  12개월 이격도: {disp_252:.1f}% (MA252: {ma_252:,.0f})')
        if disp_252 > 130:
            warnings.append(f'🔴 12개월 이격도 {disp_252:.1f}% — 역사적 극단 과열 (기준: 130%+, 과거 고점 2~4개월 선행)')
        elif disp_252 > 120:
            warnings.append(f'⚠️ 12개월 이격도 {disp_252:.1f}% — 과열 구간 진입 (기준: 120%, 중장기 고점 경고)')

    # 엘리어트 파동 위치 추정
    elliott = estimate_elliott_wave(closes)
    result['elliott'] = elliott
    print(f'  엘리어트 파동 추정: {elliott["wave_position"]}')
    if elliott['wave_risk'] == 'DANGER':
        warnings.append(f'🔴 엘리어트 파동: {elliott["wave_position"]}')
    elif elliott['wave_risk'] == 'CAUTION':
        warnings.append(f'⚠️ 엘리어트 파동: {elliott["wave_position"]}')

    result['warnings']      = warnings
    result['warning_count'] = len(warnings)
    print(f'  → 경고 {len(warnings)}건')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# AI 종합 판단 (GPT-4.1 기반 전문 리포트 생성)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ai_verdict(valuation, semiconductor, supply_demand, technical):
    print('\n━━━ [AI 종합 판단] ━━━')

    all_warnings = (
        valuation.get('warnings', []) +
        semiconductor.get('signals', []) +
        supply_demand.get('warnings', []) +
        technical.get('warnings', [])
    )
    total_warnings = len(all_warnings)
    danger_count  = sum(1 for w in all_warnings if w.startswith('🔴'))
    caution_count = sum(1 for w in all_warnings if w.startswith('⚠️'))
    watch_count   = sum(1 for w in all_warnings if w.startswith('⚡'))

    if danger_count >= 3:
        pre_risk = '🔴 위험 (DANGER)'
    elif danger_count >= 1 or caution_count >= 3:
        pre_risk = '🟠 경계 (CAUTION)'
    elif caution_count >= 1 or watch_count >= 2:
        pre_risk = '🟡 주의 (WATCH)'
    else:
        pre_risk = '🟢 정상 (NORMAL)'

    # 핵심 지표 추출
    kospi_price   = valuation.get('kospi_price', 'N/A')
    buffett_ratio = valuation.get('buffett_ratio', 'N/A')
    disp_252      = technical.get('disparity_252d') or valuation.get('disparity_252d', 'N/A')
    disp_200      = valuation.get('disparity_200d', 'N/A')
    rsi           = technical.get('rsi_14', 'N/A')
    bb_pct        = technical.get('bb_pct', 'N/A')
    rv_20         = supply_demand.get('realized_vol_20d', 'N/A')
    rv_60         = supply_demand.get('realized_vol_60d', 'N/A')
    vix           = supply_demand.get('vix', 'N/A')
    usdkrw        = supply_demand.get('usdkrw', 'N/A')
    usdkrw_3m     = supply_demand.get('usdkrw_ret_3m', 'N/A')
    ewy_3m        = supply_demand.get('ewy_ret_3m', 'N/A')
    ewy_disp      = supply_demand.get('ewy_disparity_200d', 'N/A')
    lev_vol_ratio = supply_demand.get('kodex_lev_vol_ratio', 'N/A')
    samsung_price = semiconductor.get('samsung', {}).get('price', 'N/A')
    samsung_3m    = semiconductor.get('samsung', {}).get('ret_3m', 'N/A')
    hynix_price   = semiconductor.get('hynix', {}).get('price', 'N/A')
    hynix_3m      = semiconductor.get('hynix', {}).get('ret_3m', 'N/A')
    soxx_3m       = semiconductor.get('soxx_ret_3m', 'N/A')
    nvda_3m       = semiconductor.get('nvda_ret_3m', 'N/A')
    rel_strength  = semiconductor.get('hynix_vs_samsung_rel_3m', 'N/A')
    elliott_pos   = technical.get('elliott', {}).get('wave_position', 'N/A')
    elliott_desc  = technical.get('elliott', {}).get('wave_desc', 'N/A')
    macd          = technical.get('macd', 'N/A')
    macd_hist     = technical.get('macd_histogram', 'N/A')
    samsung_fper  = valuation.get('samsung_forward_per', 'N/A')
    hynix_fper    = valuation.get('hynix_forward_per', 'N/A')
    samsung_eg    = valuation.get('samsung_earnings_growth', 'N/A')
    hynix_eg      = valuation.get('hynix_earnings_growth', 'N/A')
    samsung_mcap  = valuation.get('samsung_market_cap_trillion', 'N/A')
    hynix_mcap    = valuation.get('hynix_market_cap_trillion', 'N/A')
    pct_52w_high  = valuation.get('pct_from_52w_high', 'N/A')
    ma_align      = '정배열 ✅' if technical.get('ma_perfect_align') else '비정배열 ❌'

    summary_data = f"""
=== 코스피 중장기 고점 판독기 v2.1 — 실시간 데이터 ({datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')}) ===

[1. 밸류에이션 & 과열 지표]
- 코스피 현재가: {kospi_price:,}pt (52주 고점 대비: {pct_52w_high}%)
- 버핏 지수(GDP 대비 시가총액 추정): {buffett_ratio}%
  [경계: 120% / 주의: 150% / 역사적 과열: 180%+]
- 삼성전자 시총: {samsung_mcap}조 원 / SK하이닉스 시총: {hynix_mcap}조 원
- 12개월 이격도: {disp_252}% [과열 기준: 120% 돌파 시 2~4개월 내 고점 형성 경고]
- 200일 이격도: {disp_200}%
- 삼성전자 Forward PER: {samsung_fper}배 / 이익 성장률: {samsung_eg}
  [역사적 PER 상단: 13배. AI 리레이팅으로 기준선 상향 중이나, 이익 추정치 하향 시 고점 신호]
- SK하이닉스 Forward PER: {hynix_fper}배 / 이익 성장률: {hynix_eg}
- 경고 신호: {'; '.join(valuation.get('warnings', ['없음']))}

[2. 반도체 업황 & 수출 데이터]
- 삼성전자: {samsung_price:,}원 (3개월 수익률: {samsung_3m}%)
- SK하이닉스: {hynix_price:,}원 (3개월 수익률: {hynix_3m}%)
- 하이닉스-삼성 상대강도(3M): {rel_strength}%p [HBM 쏠림 지표, +20%p 이상 시 과도 집중]
- SOXX(필라델피아 반도체 ETF): 3개월 수익률 {soxx_3m}% [글로벌 반도체 사이클 선행 지표]
- NVIDIA: 3개월 수익률 {nvda_3m}% [HBM/AI 인프라 수요 직접 지표]
- 핵심 논리: 삼성전자·SK하이닉스의 이익 컨센서스 하향 조정이 코스피 고점보다 선행하는 경향
- 경고 신호: {'; '.join(semiconductor.get('signals', ['없음']))}

[3. 수급 & 심리 지표]
- 코스피 20일 실현변동성: {rv_20}% (60일: {rv_60}%) [연환산, VKOSPI 프록시]
  [VKOSPI 25~35 수준: 불안 / 40+ 수준: 공포 / 50+ 수준: 시장 붕괴 위험]
- VIX(미국 공포지수): {vix} [20+ 불안 / 30+ 공포]
- USD/KRW: {usdkrw:,}원 (3개월 변화: {usdkrw_3m}%) [상승 시 외국인 이탈 신호]
- EWY(외국인 수급 프록시): 3개월 수익률 {ewy_3m}% / 200일 이격도: {ewy_disp}%
- 레버리지 ETF 거래량 비율(20일/60일): {lev_vol_ratio}x [1.5x+ 시 빚투 과열]
- 경고 신호: {'; '.join(supply_demand.get('warnings', ['없음']))}

[4. 기술적 분석]
- RSI(14): {rsi} [과매수: 70+ / 극단: 80+]
- MACD: {macd} / 히스토그램: {macd_hist} [음전환 시 모멘텀 약화]
- 볼린저밴드 %B: {bb_pct}% [90%+ 극단 과열]
- 이평선 배열: {ma_align}
- 12개월 이격도: {disp_252}% [과거 사례: 120% 돌파 후 2~4개월 내 고점]
- 엘리어트 파동 위치: {elliott_pos}
- 파동 분석: {elliott_desc}
- 경고 신호: {'; '.join(technical.get('warnings', ['없음']))}

[종합 경고 집계]
- 🔴 위험 신호: {danger_count}건 / ⚠️ 주의 신호: {caution_count}건 / ⚡ 경계 신호: {watch_count}건
- 사전 위험도: {pre_risk}
- 전체 경고 목록:
{chr(10).join(f'  {w}' for w in all_warnings) if all_warnings else '  없음'}
"""

    prompt = f"""{summary_data}

당신은 삼성디스플레이 AI팀 임원에게 브리핑하는 한국 주식시장 전문 퀀트 애널리스트입니다.
위 실시간 데이터를 바탕으로 코스피 중장기 고점 위험도를 전문적으로 판단하는 HTML 리포트를 작성하세요.

핵심 분석 논리:
"단순한 지수 하락보다는 반도체 이익 전망치의 훼손(Fundamental)과 이격도 과다에 따른 기술적 부담(Technical)이 결합되는 시점을 중장기 변곡점으로 본다."

다음 형식으로 정확히 작성하세요 (HTML 태그: <b>, <i>, <br>, <hr>만 사용):

<b>🎯 코스피 중장기 고점 위험도 종합 판단</b><br>
<b>위험도: [🟢 정상 / 🟡 주의 / 🟠 경계 / 🔴 위험 중 하나 — 반드시 실제 데이터 기반으로 선택]</b><br>
[위험도 판단 근거를 2~3문장으로, 구체적 수치 인용 필수]<br>
<hr>
<b>📊 4대 프레임워크 스코어카드</b><br>
밸류에이션: [1~5점] — [핵심 수치 인용하여 한 줄 요약]<br>
반도체 업황: [1~5점] — [핵심 수치 인용하여 한 줄 요약]<br>
수급/심리: [1~5점] — [핵심 수치 인용하여 한 줄 요약]<br>
기술적 분석: [1~5점] — [핵심 수치 인용하여 한 줄 요약]<br>
<i>(1점=안전, 5점=극단 위험)</i><br>
<hr>
<b>🔍 핵심 근거 Top 3</b><br>
① [가장 중요한 신호: 구체적 수치 + 역사적 맥락 + 의미 해석, 2~3문장]<br>
② [두 번째 신호: 구체적 수치 + 역사적 맥락 + 의미 해석, 2~3문장]<br>
③ [세 번째 신호: 구체적 수치 + 역사적 맥락 + 의미 해석, 2~3문장]<br>
<hr>
<b>📈 시나리오 분석</b><br>
<b>강세 지속 시나리오:</b> [조건(예: 삼성전자 EPS 상향 유지, NVDA 실적 서프라이즈 등)과 목표 수준, 1~2문장]<br>
<b>고점 형성 시나리오:</b> [트리거 조건(예: 이격도 130% 돌파 + EPS 하향 동시 발생)과 하락 예상 폭, 1~2문장]<br>
<hr>
<b>⚡ 임원 브리핑용 모니터링 체크리스트</b><br>
① [모니터링 포인트 1: 구체적 수치 기준 포함, 언제 경보를 울릴지 명확히]<br>
② [모니터링 포인트 2: 구체적 수치 기준 포함]<br>
③ [모니터링 포인트 3: 구체적 수치 기준 포함]<br>
<hr>
<i>⚠️ 본 리포트는 AI 기반 자동 분석으로, 투자 권유가 아닙니다. 생성: {datetime.now(KST).strftime('%Y.%m.%d %H:%M KST')}</i>

규칙: 모든 항목에 실제 수치를 인용하세요. 추상적 표현 금지. 임원 브리핑 수준의 간결하고 날카로운 문장으로 작성하세요.
"""

    try:
        response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[
                {'role': 'system', 'content': '당신은 한국 주식시장 전문 퀀트 애널리스트입니다. 임원 브리핑 수준의 간결하고 날카로운 분석을 제공합니다. 모든 주장에 구체적 수치를 인용하고, 추상적 표현을 피합니다.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=2000,
            temperature=0.2,
        )
        ai_content = response.choices[0].message.content
        print(f'  → AI 판단 생성 완료 ({len(ai_content)}자)')
        return ai_content, total_warnings, all_warnings, pre_risk
    except Exception as e:
        print(f'  [AI 오류] {e}')
        fallback = f"""<b>🎯 코스피 중장기 고점 위험도 종합 판단</b><br>
<b>위험도: {pre_risk}</b><br>
총 {total_warnings}건의 경고 신호 감지 (🔴 위험 {danger_count}건 / ⚠️ 주의 {caution_count}건).<br>
<hr>
<b>📊 실시간 핵심 지표</b><br>
코스피: {kospi_price:,}pt | 버핏 지수: {buffett_ratio}% | 12개월 이격도: {disp_252}%<br>
RSI(14): {rsi} | 볼린저밴드 %B: {bb_pct}% | 20일 실현변동성: {rv_20}%<br>
삼성전자 Forward PER: {samsung_fper}배 | USD/KRW: {usdkrw:,}원 | VIX: {vix}<br>
<hr>
<b>⚠️ 감지된 경고 신호 ({total_warnings}건)</b><br>
{'<br>'.join(all_warnings) if all_warnings else '현재 주요 경고 신호 없음'}<br>
<hr>
<b>📈 엘리어트 파동 위치</b><br>
{elliott_pos}: {elliott_desc}<br>
<hr>
<i>⚠️ 본 리포트는 AI 기반 자동 분석으로, 투자 권유가 아닙니다. 생성: {datetime.now(KST).strftime('%Y.%m.%d %H:%M KST')}</i>"""
        return fallback, total_warnings, all_warnings, pre_risk


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now(KST)
    print(f'{"="*60}')
    print(f'  코스피 중장기 고점 판독기 v2.1')
    print(f'  실행 시각: {ts.strftime("%Y-%m-%dT%H:%M:%S KST")}')
    print(f'{"="*60}')

    valuation     = analyze_valuation()
    time.sleep(1)
    semiconductor = analyze_semiconductor()
    time.sleep(1)
    supply_demand = analyze_supply_demand()
    time.sleep(1)
    technical     = analyze_technical()

    ai_content, total_warnings, all_warnings, pre_risk = generate_ai_verdict(
        valuation, semiconductor, supply_demand, technical
    )

    danger_count = sum(1 for w in all_warnings if w.startswith('🔴'))
    if danger_count >= 3 or total_warnings >= 7:
        risk_level = 'DANGER'
        risk_emoji = '🔴'
    elif danger_count >= 1 or total_warnings >= 4:
        risk_level = 'CAUTION'
        risk_emoji = '🟠'
    elif total_warnings >= 2:
        risk_level = 'WATCH'
        risk_emoji = '🟡'
    else:
        risk_level = 'NORMAL'
        risk_emoji = '🟢'

    os.makedirs('reports', exist_ok=True)
    report = {
        'timestamp':      ts.isoformat(),
        'risk_level':     risk_level,
        'risk_emoji':     risk_emoji,
        'pre_risk':       pre_risk,
        'total_warnings': total_warnings,
        'warnings':       all_warnings,
        'data': {
            'valuation':     valuation,
            'semiconductor': semiconductor,
            'supply_demand': supply_demand,
            'technical':     technical,
        },
        'content': ai_content,
    }

    with open('reports/latest.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\n{"="*60}')
    print(f'✅ 분석 완료 — 코스피 중장기 고점 판독기 v2.1')
    print(f'   위험도: {risk_emoji} {risk_level} ({pre_risk})')
    print(f'   경고 신호: {total_warnings}건 (🔴 위험 {danger_count}건)')
    print(f'   → reports/latest.json 저장 완료')


if __name__ == '__main__':
    main()
