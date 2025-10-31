use algopioneer::coinbase::CoinbaseClient;
use dotenv::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from the .env file
    dotenv().ok();

    println!("--- AlgoPioneer: Initializing ---");

    // --- Connect to Coinbase ---
    println!("Connecting to Coinbase Advanced Trade API...");
    let mut client = match CoinbaseClient::new() {
        Ok(c) => {
            println!("Successfully created Coinbase client.");
            c
        },
        Err(e) => {
            eprintln!("Error creating Coinbase client: {}", e);
            return Err(e);
        }
    };

    // --- Test API Call ---
    if let Err(e) = client.test_connection().await {
        eprintln!("Connection test failed: {}", e);
        eprintln!("Please check your API key/secret, permissions, and network connection.");
        return Err(e);
    }

    println!("\n--- AlgoPioneer: Initialization Complete ---");

    Ok(())
}

