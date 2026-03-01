"""
analyze.py — 코스피 중장기 고점 판독기
=======================================
4대 핵심 프레임워크 기반 자동 분석:
  1. 밸류에이션 & 과열 지표 (Valuation & Overheat)
  2. 반도체 업황 & 수출 데이터 (Semiconductor Cycle)
  3. 수급 & 심리 지표 (Supply & Demand)
  4. 기술적 분석 (Technical Analysis)

출력: reports/latest.json (대시보드 연동)
"""

import os
import json
import time
import requests
from datetime import datetime, timezone, timedelta

# ── OpenAI 클라이언트 (AI 종합 판단) ────────────────────────────────────────
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY 환경변수 자동 사용

KST = timezone(timedelta(hours=9))
HEADERS_YFINANCE = {'User-Agent': 'Mozilla/5.0'}

# ── 유틸 함수 ────────────────────────────────────────────────────────────────

def safe_get(url, params=None, timeout=15):
    """HTTP GET 요청 (오류 시 None 반환)."""
    try:
        r = requests.get(url, params=params, headers=HEADERS_YFINANCE, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f'  [HTTP 오류] {url}: {e}')
        return None


def yf_quote(symbol):
    """Yahoo Finance 실시간 시세 조회."""
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
    data = safe_get(url, params={'interval': '1d', 'range': '1y'})
    if not data:
        return None
    try:
        result = data['chart']['result'][0]
        meta   = result['meta']
        closes = result['indicators']['quote'][0]['close']
        timestamps = result['timestamp']
        return {
            'symbol':        symbol,
            'price':         meta.get('regularMarketPrice') or meta.get('previousClose'),
            'prev_close':    meta.get('previousClose'),
            'currency':      meta.get('currency', 'KRW'),
            'closes':        [c for c in closes if c is not None],
            'timestamps':    timestamps,
        }
    except Exception as e:
        print(f'  [yf_quote 파싱 오류] {symbol}: {e}')
        return None


def yf_fundamentals(symbol):
    """Yahoo Finance 기본 재무 지표 조회."""
    url = f'https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}'
    params = {'modules': 'summaryDetail,defaultKeyStatistics,financialData'}
    data = safe_get(url, params=params)
    if not data:
        return {}
    try:
        result = data['quoteSummary']['result'][0]
        sd = result.get('summaryDetail', {})
        ks = result.get('defaultKeyStatistics', {})
        fd = result.get('financialData', {})
        return {
            'trailingPE':    sd.get('trailingPE', {}).get('raw'),
            'forwardPE':     sd.get('forwardPE', {}).get('raw'),
            'priceToBook':   sd.get('priceToBook', {}).get('raw'),
            'marketCap':     sd.get('marketCap', {}).get('raw'),
            'beta':          sd.get('beta', {}).get('raw'),
            'dividendYield': sd.get('dividendYield', {}).get('raw'),
            'eps':           ks.get('trailingEps', {}).get('raw'),
            'forwardEps':    ks.get('forwardEps', {}).get('raw'),
            'revenueGrowth': fd.get('revenueGrowth', {}).get('raw'),
            'earningsGrowth':fd.get('earningsGrowth', {}).get('raw'),
        }
    except Exception as e:
        print(f'  [yf_fundamentals 파싱 오류] {symbol}: {e}')
        return {}


# ── 프레임워크 1: 밸류에이션 & 과열 지표 ────────────────────────────────────

def analyze_valuation():
    """
    코스피 PER, PBR, 이격도, 버핏 지수(근사치) 분석.
    """
    print('\n━━━ [1] 밸류에이션 & 과열 지표 ━━━')
    result = {}

    # 코스피 지수 데이터
    kospi = yf_quote('^KS11')
    if kospi:
        closes = kospi['closes']
        current_price = kospi['price']
        result['kospi_price'] = round(current_price, 2) if current_price else None

        # 이격도: 현재가 / 12개월 이동평균 * 100
        if len(closes) >= 200:
            ma200 = sum(closes[-200:]) / 200
            disparity_200 = (current_price / ma200 * 100) if ma200 else None
            result['ma200'] = round(ma200, 2)
            result['disparity_200d'] = round(disparity_200, 2) if disparity_200 else None
            print(f'  코스피: {current_price:,.0f} / 200일 MA: {ma200:,.0f} / 이격도: {disparity_200:.1f}%')
        elif len(closes) >= 60:
            ma60 = sum(closes[-60:]) / 60
            disparity_60 = (current_price / ma60 * 100) if ma60 else None
            result['ma60'] = round(ma60, 2)
            result['disparity_60d'] = round(disparity_60, 2) if disparity_60 else None
            print(f'  코스피: {current_price:,.0f} / 60일 MA: {ma60:,.0f} / 이격도: {disparity_60:.1f}%')

        # 52주 고저
        if closes:
            result['kospi_52w_high'] = round(max(closes), 2)
            result['kospi_52w_low']  = round(min(closes), 2)
            pct_from_high = (current_price / max(closes) - 1) * 100
            result['pct_from_52w_high'] = round(pct_from_high, 2)
            print(f'  52주 고점: {max(closes):,.0f} / 현재 대비: {pct_from_high:.1f}%')

    # 삼성전자 PER/PBR (코스피 대표 종목)
    samsung_fund = yf_fundamentals('005930.KS')
    if samsung_fund:
        result['samsung_trailing_per'] = samsung_fund.get('trailingPE')
        result['samsung_forward_per']  = samsung_fund.get('forwardPE')
        result['samsung_pbr']          = samsung_fund.get('priceToBook')
        result['samsung_eps']          = samsung_fund.get('eps')
        result['samsung_forward_eps']  = samsung_fund.get('forwardEps')
        result['samsung_earnings_growth'] = samsung_fund.get('earningsGrowth')
        print(f'  삼성전자 Trailing PER: {samsung_fund.get("trailingPE")} / Forward PER: {samsung_fund.get("forwardPE")} / PBR: {samsung_fund.get("priceToBook")}')

    # SK하이닉스 PER/PBR
    hynix_fund = yf_fundamentals('000660.KS')
    if hynix_fund:
        result['hynix_trailing_per'] = hynix_fund.get('trailingPE')
        result['hynix_forward_per']  = hynix_fund.get('forwardPE')
        result['hynix_pbr']          = hynix_fund.get('priceToBook')
        result['hynix_earnings_growth'] = hynix_fund.get('earningsGrowth')
        print(f'  SK하이닉스 Trailing PER: {hynix_fund.get("trailingPE")} / Forward PER: {hynix_fund.get("forwardPE")}')

    # 코스피 ETF(KODEX 200) 기반 시장 PER 근사
    kodex200_fund = yf_fundamentals('069500.KS')
    if kodex200_fund:
        result['kospi200_per'] = kodex200_fund.get('trailingPE')
        result['kospi200_pbr'] = kodex200_fund.get('priceToBook')
        print(f'  KODEX200 PER: {kodex200_fund.get("trailingPE")} / PBR: {kodex200_fund.get("priceToBook")}')

    # 과열 판단 로직
    warnings = []
    if result.get('disparity_200d') and result['disparity_200d'] > 120:
        warnings.append(f'⚠️ 200일 이격도 {result["disparity_200d"]:.1f}% — 과열 구간 (기준: 120%)')
    elif result.get('disparity_60d') and result['disparity_60d'] > 115:
        warnings.append(f'⚠️ 60일 이격도 {result["disparity_60d"]:.1f}% — 주의 구간 (기준: 115%)')

    if result.get('samsung_forward_per') and result['samsung_forward_per'] > 15:
        warnings.append(f'⚠️ 삼성전자 Forward PER {result["samsung_forward_per"]:.1f}배 — 역사적 상단 근접')

    if result.get('kospi200_per') and result['kospi200_per'] > 13:
        warnings.append(f'⚠️ 코스피200 PER {result["kospi200_per"]:.1f}배 — 역사적 상단 (기준: 12~13배)')

    result['warnings'] = warnings
    result['warning_count'] = len(warnings)
    print(f'  → 경고 {len(warnings)}건')
    return result


# ── 프레임워크 2: 반도체 업황 & 수출 데이터 ─────────────────────────────────

def analyze_semiconductor():
    """
    삼성전자, SK하이닉스, SOXX ETF, 반도체 장비주 분석.
    """
    print('\n━━━ [2] 반도체 업황 & 수출 데이터 ━━━')
    result = {}

    # 삼성전자 주가 및 추세
    samsung = yf_quote('005930.KS')
    if samsung:
        closes = samsung['closes']
        price  = samsung['price']
        result['samsung_price'] = round(price, 0) if price else None
        if len(closes) >= 20:
            ma20 = sum(closes[-20:]) / 20
            result['samsung_ma20'] = round(ma20, 0)
            result['samsung_above_ma20'] = price > ma20 if price else None
        if len(closes) >= 60:
            ma60 = sum(closes[-60:]) / 60
            result['samsung_ma60'] = round(ma60, 0)
            result['samsung_above_ma60'] = price > ma60 if price else None
        # 3개월 수익률
        if len(closes) >= 60:
            ret_3m = (closes[-1] / closes[-60] - 1) * 100
            result['samsung_ret_3m'] = round(ret_3m, 2)
            print(f'  삼성전자: {price:,.0f}원 / 3개월 수익률: {ret_3m:.1f}%')

    # SK하이닉스 주가 및 추세
    hynix = yf_quote('000660.KS')
    if hynix:
        closes = hynix['closes']
        price  = hynix['price']
        result['hynix_price'] = round(price, 0) if price else None
        if len(closes) >= 60:
            ret_3m = (closes[-1] / closes[-60] - 1) * 100
            result['hynix_ret_3m'] = round(ret_3m, 2)
            print(f'  SK하이닉스: {price:,.0f}원 / 3개월 수익률: {ret_3m:.1f}%')

    # SOXX (필라델피아 반도체 지수 ETF) — 글로벌 반도체 사이클 선행 지표
    soxx = yf_quote('SOXX')
    if soxx:
        closes = soxx['closes']
        price  = soxx['price']
        result['soxx_price'] = round(price, 2) if price else None
        if len(closes) >= 60:
            ret_3m = (closes[-1] / closes[-60] - 1) * 100
            result['soxx_ret_3m'] = round(ret_3m, 2)
        if len(closes) >= 200:
            ma200 = sum(closes[-200:]) / 200
            result['soxx_above_ma200'] = price > ma200 if price else None
            result['soxx_ma200'] = round(ma200, 2)
            print(f'  SOXX: ${price:.2f} / 200일 MA: ${ma200:.2f} / 3개월 수익률: {ret_3m:.1f}%')

    # Nvidia (HBM/AI 인프라 수요 선행 지표)
    nvda = yf_quote('NVDA')
    if nvda:
        closes = nvda['closes']
        price  = nvda['price']
        result['nvda_price'] = round(price, 2) if price else None
        if len(closes) >= 60:
            ret_3m = (closes[-1] / closes[-60] - 1) * 100
            result['nvda_ret_3m'] = round(ret_3m, 2)
        if len(closes) >= 200:
            ma200 = sum(closes[-200:]) / 200
            result['nvda_above_ma200'] = price > ma200 if price else None
            print(f'  NVDA: ${price:.2f} / 3개월 수익률: {ret_3m:.1f}%')

    # 반도체 업황 신호
    signals = []
    if result.get('samsung_above_ma60') is False:
        signals.append('⚠️ 삼성전자 60일 이평선 하향 이탈 — 업황 둔화 신호')
    if result.get('soxx_above_ma200') is False:
        signals.append('⚠️ SOXX 200일 이평선 하향 이탈 — 글로벌 반도체 사이클 하강')
    if result.get('nvda_above_ma200') is False:
        signals.append('⚠️ NVIDIA 200일 이평선 하향 이탈 — AI 인프라 투자 둔화 우려')
    if result.get('samsung_ret_3m') is not None and result['samsung_ret_3m'] < -10:
        signals.append(f'⚠️ 삼성전자 3개월 수익률 {result["samsung_ret_3m"]:.1f}% — 급격한 하락')

    result['signals'] = signals
    result['signal_count'] = len(signals)
    print(f'  → 신호 {len(signals)}건')
    return result


# ── 프레임워크 3: 수급 & 심리 지표 ──────────────────────────────────────────

def analyze_supply_demand():
    """
    VKOSPI(공포지수), 원/달러 환율, 외국인 수급 신호 분석.
    """
    print('\n━━━ [3] 수급 & 심리 지표 ━━━')
    result = {}

    # VKOSPI (한국 변동성 지수) — Yahoo Finance에서 ^VKOSPI
    vkospi = yf_quote('^VKOSPI')
    if vkospi:
        price = vkospi['price']
        closes = vkospi['closes']
        result['vkospi'] = round(price, 2) if price else None
        if closes:
            result['vkospi_1m_avg'] = round(sum(closes[-20:]) / min(len(closes), 20), 2)
            result['vkospi_3m_avg'] = round(sum(closes[-60:]) / min(len(closes), 60), 2)
        print(f'  VKOSPI: {price:.2f} (1개월 평균: {result.get("vkospi_1m_avg", "N/A")})')

    # 원/달러 환율 (외국인 수급 프록시)
    usdkrw = yf_quote('USDKRW=X')
    if usdkrw:
        price  = usdkrw['price']
        closes = usdkrw['closes']
        result['usdkrw'] = round(price, 2) if price else None
        if len(closes) >= 60:
            ma60 = sum(closes[-60:]) / 60
            result['usdkrw_ma60'] = round(ma60, 2)
            result['usdkrw_above_ma60'] = price > ma60 if price else None
            ret_3m = (closes[-1] / closes[-60] - 1) * 100
            result['usdkrw_ret_3m'] = round(ret_3m, 2)
        print(f'  USD/KRW: {price:,.2f} / 3개월 변화: {result.get("usdkrw_ret_3m", "N/A")}%')

    # 코스피 외국인 수급 프록시: EWY (iShares MSCI South Korea ETF)
    ewy = yf_quote('EWY')
    if ewy:
        closes = ewy['closes']
        price  = ewy['price']
        result['ewy_price'] = round(price, 2) if price else None
        if len(closes) >= 20:
            ma20 = sum(closes[-20:]) / 20
            result['ewy_above_ma20'] = price > ma20 if price else None
        if len(closes) >= 60:
            ret_3m = (closes[-1] / closes[-60] - 1) * 100
            result['ewy_ret_3m'] = round(ret_3m, 2)
        if len(closes) >= 200:
            ma200 = sum(closes[-200:]) / 200
            result['ewy_above_ma200'] = price > ma200 if price else None
            result['ewy_ma200'] = round(ma200, 2)
            print(f'  EWY(외국인 수급 프록시): ${price:.2f} / 200일 MA: ${ma200:.2f}')

    # 수급 경고 신호
    warnings = []
    if result.get('vkospi') and result['vkospi'] > 30:
        warnings.append(f'🔴 VKOSPI {result["vkospi"]:.1f} — 공포 구간 (기준: 30 이상)')
    elif result.get('vkospi') and result['vkospi'] > 20:
        warnings.append(f'⚠️ VKOSPI {result["vkospi"]:.1f} — 불안 구간 (기준: 20~30)')

    if result.get('usdkrw_above_ma60') is True and result.get('usdkrw_ret_3m', 0) > 3:
        warnings.append(f'⚠️ 원/달러 환율 상승 ({result["usdkrw_ret_3m"]:.1f}%) — 외국인 자금 이탈 우려')

    if result.get('ewy_above_ma200') is False:
        warnings.append('⚠️ EWY 200일 이평선 하향 이탈 — 외국인 수급 악화 신호')

    result['warnings'] = warnings
    result['warning_count'] = len(warnings)
    print(f'  → 경고 {len(warnings)}건')
    return result


# ── 프레임워크 4: 기술적 분석 ────────────────────────────────────────────────

def analyze_technical():
    """
    코스피 RSI, MACD, 볼린저 밴드, 추세 분석.
    """
    print('\n━━━ [4] 기술적 분석 ━━━')
    result = {}

    kospi = yf_quote('^KS11')
    if not kospi or not kospi['closes']:
        print('  코스피 데이터 없음')
        return result

    closes = kospi['closes']
    price  = kospi['price']

    # RSI (14일)
    if len(closes) >= 15:
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains  = [d if d > 0 else 0 for d in deltas[-14:]]
        losses = [-d if d < 0 else 0 for d in deltas[-14:]]
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss > 0:
            rs  = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100
        result['rsi_14'] = round(rsi, 2)
        print(f'  RSI(14): {rsi:.1f}')

    # 볼린저 밴드 (20일)
    if len(closes) >= 20:
        ma20   = sum(closes[-20:]) / 20
        std20  = (sum((c - ma20) ** 2 for c in closes[-20:]) / 20) ** 0.5
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_pct   = (price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else None
        result['bb_upper'] = round(bb_upper, 2)
        result['bb_lower'] = round(bb_lower, 2)
        result['bb_pct']   = round(bb_pct, 2) if bb_pct else None
        result['ma20']     = round(ma20, 2)
        print(f'  볼린저밴드: 상단 {bb_upper:,.0f} / 하단 {bb_lower:,.0f} / %B: {bb_pct:.1f}%')

    # MACD (12, 26, 9)
    def ema(data, period):
        k = 2 / (period + 1)
        ema_val = data[0]
        for d in data[1:]:
            ema_val = d * k + ema_val * (1 - k)
        return ema_val

    if len(closes) >= 35:
        ema12  = ema(closes[-35:], 12)
        ema26  = ema(closes[-35:], 26)
        macd   = ema12 - ema26
        result['macd'] = round(macd, 2)
        result['macd_bullish'] = macd > 0
        print(f'  MACD: {macd:.2f} ({"강세" if macd > 0 else "약세"})')

    # 이동평균선 배열
    ma_signals = []
    if len(closes) >= 200:
        ma5   = sum(closes[-5:]) / 5
        ma20  = sum(closes[-20:]) / 20
        ma60  = sum(closes[-60:]) / 60
        ma120 = sum(closes[-120:]) / 120
        ma200 = sum(closes[-200:]) / 200
        result['ma5']   = round(ma5, 2)
        result['ma20']  = round(ma20, 2)
        result['ma60']  = round(ma60, 2)
        result['ma120'] = round(ma120, 2)
        result['ma200'] = round(ma200, 2)

        # 정배열 확인 (5 > 20 > 60 > 120 > 200)
        perfect_align = ma5 > ma20 > ma60 > ma120 > ma200
        result['ma_perfect_align'] = perfect_align
        print(f'  이평선 정배열: {"✅" if perfect_align else "❌"}')

        # 골든/데드 크로스 (20일 vs 60일)
        ma20_prev = sum(closes[-21:-1]) / 20
        ma60_prev = sum(closes[-61:-1]) / 60
        if ma20 > ma60 and ma20_prev <= ma60_prev:
            ma_signals.append('🟢 골든크로스 발생 (20일 > 60일)')
        elif ma20 < ma60 and ma20_prev >= ma60_prev:
            ma_signals.append('🔴 데드크로스 발생 (20일 < 60일)')

    # 기술적 경고
    tech_warnings = []
    if result.get('rsi_14') and result['rsi_14'] > 70:
        tech_warnings.append(f'⚠️ RSI {result["rsi_14"]:.1f} — 과매수 구간 (기준: 70 이상)')
    elif result.get('rsi_14') and result['rsi_14'] < 30:
        tech_warnings.append(f'🟢 RSI {result["rsi_14"]:.1f} — 과매도 구간 (반등 기회)')

    if result.get('bb_pct') and result['bb_pct'] > 90:
        tech_warnings.append(f'⚠️ 볼린저밴드 %B {result["bb_pct"]:.1f}% — 상단 돌파 과열')

    if result.get('macd_bullish') is False:
        tech_warnings.append('⚠️ MACD 음전환 — 단기 하락 모멘텀')

    result['warnings']     = tech_warnings
    result['ma_signals']   = ma_signals
    result['warning_count'] = len(tech_warnings)
    print(f'  → 경고 {len(tech_warnings)}건')
    return result


# ── AI 종합 판단 ─────────────────────────────────────────────────────────────

def generate_ai_verdict(valuation, semiconductor, supply_demand, technical):
    """
    4대 프레임워크 분석 결과를 종합하여 AI가 고점 위험도를 판단.
    """
    print('\n━━━ [AI 종합 판단] ━━━')

    # 경고 신호 집계
    all_warnings = (
        valuation.get('warnings', []) +
        semiconductor.get('signals', []) +
        supply_demand.get('warnings', []) +
        technical.get('warnings', [])
    )
    total_warnings = len(all_warnings)

    # 데이터 요약 텍스트 구성
    summary_data = f"""
=== 코스피 고점 판독기 — 실시간 데이터 요약 ({datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')}) ===

[1. 밸류에이션 & 과열 지표]
- 코스피 현재가: {valuation.get('kospi_price', 'N/A'):,}
- 52주 고점 대비: {valuation.get('pct_from_52w_high', 'N/A')}%
- 200일 이격도: {valuation.get('disparity_200d', valuation.get('disparity_60d', 'N/A'))}%
- 삼성전자 Trailing PER: {valuation.get('samsung_trailing_per', 'N/A')}배
- 삼성전자 Forward PER: {valuation.get('samsung_forward_per', 'N/A')}배
- 삼성전자 PBR: {valuation.get('samsung_pbr', 'N/A')}배
- SK하이닉스 Forward PER: {valuation.get('hynix_forward_per', 'N/A')}배
- KODEX200 PER: {valuation.get('kospi200_per', 'N/A')}배
- 경고 신호: {', '.join(valuation.get('warnings', ['없음']))}

[2. 반도체 업황 & 수출 데이터]
- 삼성전자: {valuation.get('samsung_price', semiconductor.get('samsung_price', 'N/A'))}원 (3개월 수익률: {semiconductor.get('samsung_ret_3m', 'N/A')}%)
- SK하이닉스: {semiconductor.get('hynix_price', 'N/A')}원 (3개월 수익률: {semiconductor.get('hynix_ret_3m', 'N/A')}%)
- SOXX ETF: ${semiconductor.get('soxx_price', 'N/A')} (3개월 수익률: {semiconductor.get('soxx_ret_3m', 'N/A')}%, 200일 MA 위: {semiconductor.get('soxx_above_ma200', 'N/A')})
- NVIDIA: ${semiconductor.get('nvda_price', 'N/A')} (3개월 수익률: {semiconductor.get('nvda_ret_3m', 'N/A')}%, 200일 MA 위: {semiconductor.get('nvda_above_ma200', 'N/A')})
- 경고 신호: {', '.join(semiconductor.get('signals', ['없음']))}

[3. 수급 & 심리 지표]
- VKOSPI(공포지수): {supply_demand.get('vkospi', 'N/A')} (1개월 평균: {supply_demand.get('vkospi_1m_avg', 'N/A')})
- USD/KRW: {supply_demand.get('usdkrw', 'N/A')} (3개월 변화: {supply_demand.get('usdkrw_ret_3m', 'N/A')}%)
- EWY(외국인 수급): ${supply_demand.get('ewy_price', 'N/A')} (200일 MA 위: {supply_demand.get('ewy_above_ma200', 'N/A')}, 3개월 수익률: {supply_demand.get('ewy_ret_3m', 'N/A')}%)
- 경고 신호: {', '.join(supply_demand.get('warnings', ['없음']))}

[4. 기술적 분석]
- RSI(14): {technical.get('rsi_14', 'N/A')}
- MACD: {technical.get('macd', 'N/A')} (강세: {technical.get('macd_bullish', 'N/A')})
- 볼린저밴드 %B: {technical.get('bb_pct', 'N/A')}%
- 이평선 정배열: {technical.get('ma_perfect_align', 'N/A')}
- 이평선 신호: {', '.join(technical.get('ma_signals', ['없음']))}
- 경고 신호: {', '.join(technical.get('warnings', ['없음']))}

[종합 경고 집계]
- 총 경고/신호 수: {total_warnings}건
- 상세: {chr(10).join(all_warnings) if all_warnings else '없음'}
"""

    prompt = f"""{summary_data}

당신은 한국 주식시장 전문 퀀트 애널리스트입니다. 위의 실시간 데이터를 바탕으로 코스피 중장기 고점 위험도를 판단해주세요.

다음 형식으로 한국어 HTML 리포트를 작성하세요:

<b>🎯 코스피 고점 위험도 종합 판단</b><br>
[위험도: 🟢 낮음 / 🟡 주의 / 🟠 경계 / 🔴 위험 중 하나 선택]<br>
<br>
<b>📊 4대 프레임워크 스코어카드</b><br>
[각 프레임워크별 1~5점 위험도 점수와 한 줄 요약]<br>
<br>
<b>🔍 핵심 근거</b><br>
[가장 중요한 3가지 신호를 구체적 수치와 함께 설명]<br>
<br>
<b>📈 시나리오 분석</b><br>
[강세 지속 시나리오 vs 고점 형성 시나리오 각 1~2문장]<br>
<br>
<b>⚡ 투자자 체크리스트</b><br>
[지금 당장 확인해야 할 3가지 핵심 모니터링 포인트]<br>
<br>
[마지막 줄: 생성 시각 및 면책 조항]

HTML 태그는 <b>, <i>, <br>, <hr>만 사용하세요. 분석은 객관적이고 데이터 기반으로 작성하세요.
"""

    try:
        response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[
                {'role': 'system', 'content': '당신은 한국 주식시장 전문 퀀트 애널리스트입니다. 데이터 기반의 객관적이고 전문적인 분석을 제공합니다.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=1500,
            temperature=0.3,
        )
        ai_content = response.choices[0].message.content
        print(f'  → AI 판단 생성 완료 ({len(ai_content)}자)')
        return ai_content, total_warnings, all_warnings
    except Exception as e:
        print(f'  [AI 오류] {e}')
        # AI 실패 시 기본 요약 반환
        fallback = f"""<b>🎯 코스피 고점 판독기 — 데이터 요약</b><br>
<br>
<b>📊 실시간 지표</b><br>
코스피: {valuation.get('kospi_price', 'N/A'):,}pt | VKOSPI: {supply_demand.get('vkospi', 'N/A')} | RSI(14): {technical.get('rsi_14', 'N/A')}<br>
삼성전자 Forward PER: {valuation.get('samsung_forward_per', 'N/A')}배 | USD/KRW: {supply_demand.get('usdkrw', 'N/A')}<br>
<br>
<b>⚠️ 경고 신호 ({total_warnings}건)</b><br>
{'<br>'.join(all_warnings) if all_warnings else '현재 주요 경고 신호 없음'}<br>
<br>
🤖 생성: {datetime.now(KST).strftime('%Y.%m.%d %H:%M KST')} | KOSPI Peak Detector"""
        return fallback, total_warnings, all_warnings


# ── 메인 실행 ────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now(KST)
    print(f'{"="*55}')
    print(f'  코스피 고점 판독기')
    print(f'  실행 시각: {ts.strftime("%Y-%m-%dT%H:%M:%S KST")}')
    print(f'{"="*55}')

    # 4대 프레임워크 분석
    valuation     = analyze_valuation()
    time.sleep(1)  # API 레이트 리밋 방지
    semiconductor = analyze_semiconductor()
    time.sleep(1)
    supply_demand = analyze_supply_demand()
    time.sleep(1)
    technical     = analyze_technical()

    # AI 종합 판단
    ai_content, total_warnings, all_warnings = generate_ai_verdict(
        valuation, semiconductor, supply_demand, technical
    )

    # 위험도 레벨 결정
    if total_warnings >= 5:
        risk_level = 'DANGER'
        risk_emoji = '🔴'
    elif total_warnings >= 3:
        risk_level = 'CAUTION'
        risk_emoji = '🟠'
    elif total_warnings >= 1:
        risk_level = 'WATCH'
        risk_emoji = '🟡'
    else:
        risk_level = 'NORMAL'
        risk_emoji = '🟢'

    # 리포트 저장
    os.makedirs('reports', exist_ok=True)
    report = {
        'timestamp': ts.isoformat(),
        'risk_level': risk_level,
        'risk_emoji': risk_emoji,
        'total_warnings': total_warnings,
        'warnings': all_warnings,
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

    print(f'\n{"="*55}')
    print(f'✅ 분석 완료')
    print(f'   위험도: {risk_emoji} {risk_level}')
    print(f'   경고 신호: {total_warnings}건')
    print(f'   → reports/latest.json 저장')


if __name__ == '__main__':
    main()
