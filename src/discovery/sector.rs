use lazy_static::lazy_static;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sector {
    L1,             // Layer 1 (BTC, ETH, SOL, ADA)
    DeFi,           // DeFi (UNI, AAVE, MKR, COMP, CRV)
    Meme,           // Meme Coins (DOGE, SHIB)
    Infrastructure, // Infra/Oracle (LINK, GRT, FIL)
    Exchange,       // Exchange tokens (CRO, BNB - maybe not on Coinbase)
    Payment,        // Payments (XRP, XLM, LTC, BCH)
    L2,             // Layer 2 (OP, ARB, MATIC)
    Metaverse,      // Gaming/Metaverse (AXS, SAND, MANA)
    Stablecoin,     // USDT, USDC (usually excluded but good for basis)
    Unknown,
}

lazy_static! {
    static ref TOKEN_SECTORS: HashMap<&'static str, Sector> = {
        let mut m = HashMap::new();
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
        m.insert("FLOW", Sector::L1); // Specialized L1
        m.insert("SUI", Sector::L1);
        m.insert("SEI", Sector::L1);
        m.insert("APT", Sector::L1);

        // L2s
        m.insert("MATIC", Sector::L2); // Polygon (Sidechain/L2)
        m.insert("OP", Sector::L2);
        m.insert("ARB", Sector::L2);
        m.insert("IMX", Sector::L2);
        m.insert("STX", Sector::L2); // Bitcoin L2

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
        m.insert("DOGE", Sector::Meme); // Also payment
        // Note: DOGE is technically payment but moves like Meme

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
        m.insert("VET", Sector::Infrastructure); // Supply chain infra

        // Metaverse/Gaming
        m.insert("AXS", Sector::Metaverse);
        m.insert("SAND", Sector::Metaverse);
        m.insert("MANA", Sector::Metaverse);
        m.insert("GALA", Sector::Metaverse);
        m.insert("APE", Sector::Metaverse);

        m
    };
}

pub fn get_sector(symbol: &str) -> Sector {
    // Strip "-USD" suffix if present
    let ticker = symbol.split('-').next().unwrap_or(symbol);
    *TOKEN_SECTORS.get(ticker).unwrap_or(&Sector::Unknown)
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
