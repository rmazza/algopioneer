use lazy_static::lazy_static;
use std::collections::HashMap;
use tracing::debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sector {
    // Crypto Sectors
    L1,             // Layer 1 (BTC, ETH, SOL, ADA)
    DeFi,           // DeFi (UNI, AAVE, MKR, COMP, CRV)
    Meme,           // Meme Coins (DOGE, SHIB)
    Infrastructure, // Infra/Oracle (LINK, GRT, FIL)
    Exchange,       // Exchange tokens (CRO, BNB - maybe not on Coinbase)
    Payment,        // Payments (XRP, XLM, LTC, BCH)
    L2,             // Layer 2 (OP, ARB, MATIC)
    Metaverse,      // Gaming/Metaverse (AXS, SAND, MANA)
    Stablecoin,     // USDT, USDC (usually excluded but good for basis)

    // Equity Sectors (GICS-based)
    ConsumerStaples, // KO, PEP, PG, CL
    Financials,      // JPM, BAC, V, MA, GS
    Energy,          // XOM, CVX, COP
    Technology,      // MSFT, ORCL, INTC, AMD
    Healthcare,      // JNJ, PFE, MRK, UNH
    Retail,          // WMT, TGT, COST, HD
    Industrials,     // CAT, BA, GE
    Airlines,        // DAL, UAL, AAL, LUV
    Telecom,         // T, VZ
    Automotive,      // F, GM, TSLA

    Unknown,
}

lazy_static! {
    static ref TOKEN_SECTORS: HashMap<&'static str, Sector> = {
        let mut m = HashMap::new();

        // === CRYPTO SECTORS ===

        // L1s
        m.insert("BTC", Sector::L1);
        m.insert("ETH", Sector::L1);
        m.insert("SOL", Sector::L1);
        m.insert("ADA", Sector::L1);
        m.insert("DOT", Sector::L1);
        m.insert("AVAX", Sector::L1);
        m.insert("ATOM", Sector::L1);
        m.insert("NEAR", Sector::L1);
        m.insert("ALGO", Sector::L1);
        m.insert("HBAR", Sector::L1);
        m.insert("EGLD", Sector::L1);
        m.insert("EOS", Sector::L1);
        m.insert("XTZ", Sector::L1);
        m.insert("FLOW", Sector::L1);
        m.insert("SUI", Sector::L1);
        m.insert("SEI", Sector::L1);
        m.insert("APT", Sector::L1);

        // L2s
        m.insert("MATIC", Sector::L2);
        m.insert("OP", Sector::L2);
        m.insert("ARB", Sector::L2);
        m.insert("IMX", Sector::L2);
        m.insert("STX", Sector::L2);

        // DeFi
        m.insert("UNI", Sector::DeFi);
        m.insert("AAVE", Sector::DeFi);
        m.insert("MKR", Sector::DeFi);
        m.insert("COMP", Sector::DeFi);
        m.insert("CRV", Sector::DeFi);
        m.insert("SNX", Sector::DeFi);
        m.insert("LDO", Sector::DeFi);
        m.insert("RPL", Sector::DeFi);
        m.insert("ENS", Sector::DeFi);
        m.insert("1INCH", Sector::DeFi);

        // Payments / Old Gen
        m.insert("XRP", Sector::Payment);
        m.insert("XLM", Sector::Payment);
        m.insert("LTC", Sector::Payment);
        m.insert("BCH", Sector::Payment);
        m.insert("DOGE", Sector::Meme);

        // Meme
        m.insert("SHIB", Sector::Meme);
        m.insert("PEPE", Sector::Meme);
        m.insert("BONK", Sector::Meme);
        m.insert("FLOKI", Sector::Meme);

        // Infrastructure
        m.insert("LINK", Sector::Infrastructure);
        m.insert("GRT", Sector::Infrastructure);
        m.insert("FIL", Sector::Infrastructure);
        m.insert("RNDR", Sector::Infrastructure);
        m.insert("FET", Sector::Infrastructure);
        m.insert("TIA", Sector::Infrastructure);
        m.insert("VET", Sector::Infrastructure);

        // Metaverse/Gaming
        m.insert("AXS", Sector::Metaverse);
        m.insert("SAND", Sector::Metaverse);
        m.insert("MANA", Sector::Metaverse);
        m.insert("GALA", Sector::Metaverse);
        m.insert("APE", Sector::Metaverse);

        // === EQUITY SECTORS ===

        // Consumer Staples
        m.insert("KO", Sector::ConsumerStaples);
        m.insert("PEP", Sector::ConsumerStaples);
        m.insert("PG", Sector::ConsumerStaples);
        m.insert("CL", Sector::ConsumerStaples);
        m.insert("KHC", Sector::ConsumerStaples);
        m.insert("GIS", Sector::ConsumerStaples);
        m.insert("MO", Sector::ConsumerStaples);
        m.insert("PM", Sector::ConsumerStaples);
        m.insert("KMB", Sector::ConsumerStaples);

        // Financials - Banks
        m.insert("JPM", Sector::Financials);
        m.insert("BAC", Sector::Financials);
        m.insert("WFC", Sector::Financials);
        m.insert("C", Sector::Financials);
        m.insert("GS", Sector::Financials);
        m.insert("MS", Sector::Financials);
        // Financials - Payment Networks
        m.insert("V", Sector::Financials);
        m.insert("MA", Sector::Financials);
        m.insert("AXP", Sector::Financials);
        m.insert("PYPL", Sector::Financials);

        // Energy
        m.insert("XOM", Sector::Energy);
        m.insert("CVX", Sector::Energy);
        m.insert("COP", Sector::Energy);
        m.insert("OXY", Sector::Energy);
        m.insert("EOG", Sector::Energy);
        m.insert("SLB", Sector::Energy);
        m.insert("HAL", Sector::Energy);

        // Technology
        m.insert("MSFT", Sector::Technology);
        m.insert("ORCL", Sector::Technology);
        m.insert("CRM", Sector::Technology);
        m.insert("IBM", Sector::Technology);
        m.insert("INTC", Sector::Technology);
        m.insert("AMD", Sector::Technology);
        m.insert("NVDA", Sector::Technology);
        m.insert("AVGO", Sector::Technology);
        m.insert("QCOM", Sector::Technology);
        m.insert("TXN", Sector::Technology);
        m.insert("AAPL", Sector::Technology);
        m.insert("GOOGL", Sector::Technology);
        m.insert("META", Sector::Technology);

        // Healthcare
        m.insert("JNJ", Sector::Healthcare);
        m.insert("PFE", Sector::Healthcare);
        m.insert("MRK", Sector::Healthcare);
        m.insert("ABBV", Sector::Healthcare);
        m.insert("UNH", Sector::Healthcare);
        m.insert("LLY", Sector::Healthcare);
        m.insert("BMY", Sector::Healthcare);
        m.insert("AMGN", Sector::Healthcare);

        // Retail
        m.insert("WMT", Sector::Retail);
        m.insert("TGT", Sector::Retail);
        m.insert("COST", Sector::Retail);
        m.insert("HD", Sector::Retail);
        m.insert("LOW", Sector::Retail);
        m.insert("AMZN", Sector::Retail);
        m.insert("EBAY", Sector::Retail);

        // Industrials
        m.insert("CAT", Sector::Industrials);
        m.insert("BA", Sector::Industrials);
        m.insert("GE", Sector::Industrials);
        m.insert("HON", Sector::Industrials);
        m.insert("UPS", Sector::Industrials);
        m.insert("FDX", Sector::Industrials);
        m.insert("MMM", Sector::Industrials);

        // Airlines
        m.insert("DAL", Sector::Airlines);
        m.insert("UAL", Sector::Airlines);
        m.insert("AAL", Sector::Airlines);
        m.insert("LUV", Sector::Airlines);
        m.insert("JBLU", Sector::Airlines);

        // Telecom
        m.insert("T", Sector::Telecom);
        m.insert("VZ", Sector::Telecom);
        m.insert("TMUS", Sector::Telecom);

        // Automotive
        m.insert("F", Sector::Automotive);
        m.insert("GM", Sector::Automotive);
        m.insert("TSLA", Sector::Automotive);
        m.insert("RIVN", Sector::Automotive);

        m
    };
}

pub fn get_sector(symbol: &str) -> Sector {
    // Strip "-USD" suffix if present (for crypto)
    let ticker = symbol.split('-').next().unwrap_or(symbol);
    match TOKEN_SECTORS.get(ticker) {
        Some(sector) => *sector,
        None => {
            debug!(
                symbol = %symbol,
                ticker = %ticker,
                "Unknown sector mapping - symbol not in classification table"
            );
            Sector::Unknown
        }
    }
}

/// Check if two symbols belong to the same sector
pub fn is_same_sector(a: &str, b: &str) -> bool {
    let sector_a = get_sector(a);
    let sector_b = get_sector(b);

    // Unknowns never match
    if sector_a == Sector::Unknown || sector_b == Sector::Unknown {
        return false;
    }

    sector_a == sector_b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_sectors() {
        assert_eq!(get_sector("BTC-USD"), Sector::L1);
        assert_eq!(get_sector("AAVE-USD"), Sector::DeFi);
        assert_eq!(get_sector("SAND"), Sector::Metaverse);
    }

    #[test]
    fn test_equity_sectors() {
        assert_eq!(get_sector("KO"), Sector::ConsumerStaples);
        assert_eq!(get_sector("PEP"), Sector::ConsumerStaples);
        assert!(is_same_sector("KO", "PEP"));

        assert_eq!(get_sector("V"), Sector::Financials);
        assert_eq!(get_sector("MA"), Sector::Financials);
        assert!(is_same_sector("V", "MA"));

        assert_eq!(get_sector("XOM"), Sector::Energy);
        assert_eq!(get_sector("CVX"), Sector::Energy);
        assert!(is_same_sector("XOM", "CVX"));
    }

    #[test]
    fn test_cross_sector_no_match() {
        assert!(!is_same_sector("KO", "XOM")); // Consumer vs Energy
        assert!(!is_same_sector("BTC", "AAPL")); // Crypto vs Equity
    }
}
