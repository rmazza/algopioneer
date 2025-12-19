use algopioneer::discovery::sector::{get_sector, is_same_sector};

fn main() {
    let pairs = vec![
        ("FLOW-USD", "VET-USD"),
        ("FLOW-USD", "DOT-USD"),
        ("MKR-USD", "SHIB-USD"),
        ("MKR-USD", "COMP-USD"),
        ("STX-USD", "LDO-USD"),
        ("DOGE-USD", "LDO-USD"),
        ("STX-USD", "AXS-USD"),
        ("MKR-USD", "XRP-USD"),
        ("AAVE-USD", "XLM-USD"),
        ("EOS-USD", "AXS-USD"),
    ];

    println!(
        "{:<25} | {:<15} | {:<15} | {}",
        "Pair", "Sector A", "Sector B", "Same?"
    );
    println!("{}", "-".repeat(70));

    for (a, b) in pairs {
        let sec_a = get_sector(a);
        let sec_b = get_sector(b);
        let same = is_same_sector(a, b);

        println!(
            "{:<25} | {:<15?} | {:<15?} | {}",
            format!("{}/{}", a, b),
            sec_a,
            sec_b,
            if same { "✅ YES" } else { "❌ NO" }
        );
    }
}
