use std::fs::OpenOptions;
use std::io::Write;

pub fn save_trade(trade_details: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("sandbox_trades.csv")?;

    // Add a newline if the file is not empty
    if file.metadata()?.len() > 0 {
        writeln!(file, "")?;
    }

    writeln!(file, "{}", trade_details)?;
    println!("-- Sandbox Mode: Trade saved to sandbox_trades.csv --");
    Ok(())
}
