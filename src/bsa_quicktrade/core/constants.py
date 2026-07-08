"""NSE F&O stock universe, index tickers, and sector mappings.

The definitive F&O list is fetched dynamically via ``nsepython.fnolist()``
at runtime.  A hardcoded fallback is maintained so the framework still
works when the NSE endpoint is unreachable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Index Tickers (yfinance format) ─────────────────────────────────────────

INDEX_TICKERS: dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY FINANCIAL SERVICES": "^CNXFIN",
    "NIFTY MIDCAP 50": "^NSEMDCP50",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY PSE": "^CNXPSE",
}

# ── Sector Mappings ─────────────────────────────────────────────────────────

SECTOR_MAP: dict[str, str] = {
    # Banking & Financial
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BANKBARODA": "Banking", "PNB": "Banking", "FEDERALBNK": "Banking",
    "IDFCFIRSTB": "Banking", "AUBANK": "Banking", "BANDHANBNK": "Banking",
    "CUB": "Banking", "RBLBANK": "Banking", "UNIONBANK": "Banking",
    "BAJFINANCE": "Financial Services", "BAJAJFINSV": "Financial Services",
    "SBILIFE": "Financial Services", "HDFCLIFE": "Financial Services",
    "HDFCAMC": "Financial Services", "SBICARD": "Financial Services",
    "CHOLAFIN": "Financial Services", "MUTHOOTFIN": "Financial Services",
    "MANAPPURAM": "Financial Services", "SHRIRAMFIN": "Financial Services",
    "LICHSGFIN": "Financial Services", "CANFINHOME": "Financial Services",
    "PFC": "Financial Services", "RECLTD": "Financial Services",
    "MFSL": "Financial Services", "ABCAPITAL": "Financial Services",
    "MCX": "Financial Services", "IEX": "Financial Services",

    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT", "LTTS": "IT", "MPHASIS": "IT",
    "COFORGE": "IT", "PERSISTENT": "IT", "TATAELXSI": "IT",
    "NAUKRI": "IT", "OFSS": "IT",

    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "APOLLOHOSP": "Healthcare", "LUPIN": "Pharma",
    "AUROPHARMA": "Pharma", "BIOCON": "Pharma", "TORNTPHARM": "Pharma",
    "ALKEM": "Pharma", "IPCALAB": "Pharma", "LALPATHLAB": "Healthcare",
    "METROPOLIS": "Healthcare", "SYNGENE": "Pharma", "GRANULES": "Pharma",
    "LAURUSLABS": "Pharma", "GLENMARK": "Pharma", "ZYDUSLIFE": "Pharma",
    "ABBOTINDIA": "Pharma",

    # Auto
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto",
    "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "TVSMOTOR": "Auto", "ASHOKLEY": "Auto", "MOTHERSON": "Auto",
    "BALKRISIND": "Auto", "ESCORTS": "Auto", "EXIDEIND": "Auto",
    "BOSCHLTD": "Auto",

    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "DABUR": "FMCG",
    "MARICO": "FMCG", "GODREJCP": "FMCG", "COLPAL": "FMCG",
    "UBL": "FMCG", "MCDOWELL-N": "FMCG", "UNITDSPR": "FMCG",
    "DEVYANI": "FMCG", "JUBLFOOD": "FMCG", "PVRINOX": "FMCG",

    # Metal & Mining
    "TATASTEEL": "Metal", "JSWSTEEL": "Metal", "HINDALCO": "Metal",
    "VEDL": "Metal", "SAIL": "Metal", "JINDALSTEL": "Metal",
    "NATIONALUM": "Metal", "HINDCOPPER": "Metal", "NMDC": "Metal",
    "COALINDIA": "Metal",

    # Energy & Oil
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "HINDPETRO": "Energy", "GAIL": "Energy",
    "PETRONET": "Energy", "NTPC": "Power", "POWERGRID": "Power",
    "TATAPOWER": "Power", "TORNTPOWER": "Power", "ADANIPORTS": "Infra",

    # Infra & Capital Goods
    "LT": "Infra", "ADANIENT": "Infra", "ABB": "Capital Goods",
    "SIEMENS": "Capital Goods", "HAL": "Defence", "BEL": "Defence",
    "BHEL": "Capital Goods", "CUMMINSIND": "Capital Goods",
    "CROMPTON": "Capital Goods", "HAVELLS": "Capital Goods",
    "POLYCAB": "Capital Goods", "VOLTAS": "Capital Goods",

    # Real Estate
    "DLF": "Real Estate", "GODREJPROP": "Real Estate",
    "OBEROIRLTY": "Real Estate", "PRESTIGE": "Real Estate",

    # Cement
    "ULTRACEMCO": "Cement", "AMBUJACEM": "Cement", "GRASIM": "Cement",
    "DALBHARAT": "Cement", "RAMCOCEM": "Cement", "JKCEMENT": "Cement",

    # Telecom & Media
    "BHARTIARTL": "Telecom", "ZEEL": "Media", "SUNTV": "Media",
    "STAR": "Media", "TATACOMM": "Telecom",

    # Chemicals
    "PIDILITIND": "Chemicals", "SRF": "Chemicals", "DEEPAKNTR": "Chemicals",
    "ATUL": "Chemicals", "PIIND": "Chemicals", "NAVINFLUOR": "Chemicals",
    "CHAMBLFERT": "Chemicals", "COROMANDEL": "Chemicals", "GNFC": "Chemicals",

    # Consumer Durables
    "TITAN": "Consumer Durables", "ASIANPAINT": "Consumer Durables",
    "BERGEPAINT": "Consumer Durables", "BATAINDIA": "Consumer Durables",
    "PAGEIND": "Consumer Durables", "DIXON": "Consumer Durables",
    "WHIRLPOOL": "Consumer Durables", "HONAUT": "Consumer Durables",
    "TRENT": "Consumer Durables",

    # Others
    "IRCTC": "Travel", "INDIGO": "Aviation", "CONCOR": "Logistics",
    "INDIAMART": "E-Commerce", "ASTRAL": "Building Materials",
    "TATACHEM": "Chemicals", "ABFRL": "Retail",
    "GSPL": "Gas Distribution", "GUJGASLTD": "Gas Distribution",
    "MGL": "Gas Distribution", "FSL": "IT", "NBCC": "Infra", "NCC": "Infra",
    "GMRINFRA": "Infra", "PEL": "Financial Services",
}


# ── Hardcoded F&O Fallback List ─────────────────────────────────────────────

HARDCODED_FNO_SYMBOLS: list[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC", "LT", "AXISBANK",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH", "TITAN",
    "SUNPHARMA", "ULTRACEMCO", "WIPRO", "NESTLEIND", "TATAMOTORS",
    "TATASTEEL", "NTPC", "POWERGRID", "M&M", "TECHM", "BAJAJFINSV",
    "ADANIENT", "ADANIPORTS", "JSWSTEEL", "ONGC", "COALINDIA",
    "DRREDDY", "CIPLA", "EICHERMOT", "APOLLOHOSP", "DIVISLAB",
    "GRASIM", "BPCL", "BRITANNIA", "BAJAJ-AUTO", "HEROMOTOCO",
    "INDUSINDBK", "TATACONSUM", "HINDALCO", "SBILIFE", "HDFCLIFE",
    "UPL", "SHRIRAMFIN",
    # Extended F&O list
    "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ALKEM", "AMBUJACEM",
    "AUROPHARMA", "ASTRAL", "ATUL", "AUBANK", "BALKRISIND",
    "BANDHANBNK", "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT",
    "BHEL", "BIOCON", "BOSCHLTD", "CANFINHOME", "CHAMBLFERT",
    "CHOLAFIN", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL",
    "CROMPTON", "CUB", "CUMMINSIND", "DABUR", "DALBHARAT",
    "DEEPAKNTR", "DELTACORP", "DEVYANI", "DIXON", "DLF",
    "ESCORTS", "EXIDEIND", "FEDERALBNK", "FSL", "GAIL",
    "GLENMARK", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP",
    "GRANULES", "GSPL", "GUJGASLTD", "HAL", "HAVELLS",
    "HDFCAMC", "HINDCOPPER", "HINDPETRO", "HONAUT",
    "IDFCFIRSTB", "IEX", "INDHOTEL", "INDIACEM",
    "INDIAMART", "INDIGO", "IOC", "IPCALAB", "IRCTC",
    "JINDALSTEL", "JKCEMENT", "JUBLFOOD", "LALPATHLAB", "LAURUSLABS",
    "LICHSGFIN", "LTIM", "LTTS", "LUPIN", "MANAPPURAM",
    "MARICO", "MCDOWELL-N", "MCX", "METROPOLIS", "MFSL",
    "MGL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN",
    "NAM-INDIA", "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NBCC",
    "NCC", "NMDC", "OBEROIRLTY", "OFSS", "PAGEIND",
    "PEL", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND",
    "PIIND", "PNB", "POLYCAB", "PRESTIGE", "PVRINOX",
    "RAMCOCEM", "RBLBANK", "RECLTD", "SAIL", "SBICARD",
    "SIEMENS", "SRF", "STAR", "SUNTV", "SYNGENE",
    "TATACHEM", "TATACOMM", "TATAELXSI", "TATAPOWER", "TORNTPHARM",
    "TORNTPOWER", "TRENT", "TVSMOTOR", "UBL", "UNIONBANK",
    "UNITDSPR", "VEDL", "VOLTAS", "WHIRLPOOL", "ZEEL", "ZYDUSLIFE",
]


def get_fno_symbols() -> list[str]:
    """Fetch the current F&O symbol list from NSE, falling back to hardcoded."""
    try:
        from nsepython import fnolist  # type: ignore[import-untyped]

        symbols: list[str] = fnolist()
        if symbols and len(symbols) > 50:
            logger.info("Fetched %d F&O symbols from NSE", len(symbols))
            return symbols
    except Exception as exc:
        logger.warning("nsepython.fnolist() failed (%s), using fallback", exc)

    logger.info("Using hardcoded F&O list (%d symbols)", len(HARDCODED_FNO_SYMBOLS))
    return list(HARDCODED_FNO_SYMBOLS)


def to_yfinance_ticker(symbol: str) -> str:
    """Convert an NSE symbol to the yfinance ticker format."""
    return f"{symbol}.NS"


def get_sector(symbol: str) -> str:
    """Return the sector for a given NSE symbol."""
    clean = symbol.replace(".NS", "")
    return SECTOR_MAP.get(clean, "Unknown")
