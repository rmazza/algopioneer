// Use the correct crate name 'cbadv'
use cbadv::{RestClient, RestClientBuilder}; // Import Coinbase client components
use dotenv::dotenv; // For loading .env file
use std::env; // For reading environment variables

// Use the tokio::main macro to set up the async runtime
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from the .env file
    dotenv().ok(); // .ok() ignores the error if .env is not found (useful for deployment)

    println!("--- AlgoPioneer: Initializing ---");

    // Retrieve API Key and Secret from environment variables
    let _api_key = env::var("COINBASE_API_KEY")
        .expect("COINBASE_API_KEY must be set in .env file or environment.");
    let _api_secret = env::var("COINBASE_API_SECRET")
        .expect("COINBASE_API_SECRET must be set in .env file or environment.");

    // --- Connect to Coinbase ---
    println!("Connecting to Coinbase Advanced Trade API...");

    // Build the REST client using the cbadv builder pattern
        let mut client: RestClient = match RestClientBuilder::new().build() {
        Ok(c) => {
            println!("Successfully created Coinbase client.");
            c
        },
        Err(e) => {
            eprintln!("Error creating Coinbase client: {}", e);
            // Return the error, converting it to a dynamic error type
                return Err(Box::new(e) as Box<dyn std::error::Error>);
        }
    };

    // --- Test API Call: Get Server Time ---
    println!("Attempting a test API call (get server time)...");

        match client.public.time().await {
        Ok(server_time) => {
            println!("Successfully connected!");
            println!("Coinbase Server Time (ISO): {}", server_time.iso);
            println!("Coinbase Server Time (Epoch Seconds): {}", server_time.epoch_seconds);
        },
        Err(e) => {
            eprintln!("Error during test API call: {}", e);
            eprintln!("Please check your API key/secret, permissions, and network connection.");
             // Return the error
                return Err(Box::new(e) as Box<dyn std::error::Error>);
        }
    }

    println!("\n--- AlgoPioneer: Initialization Complete ---");

    // We will add data fetching and strategy logic here in the next steps...

    Ok(()) // Return Ok to indicate successful execution
}

