use cbadv::{RestClient, RestClientBuilder};
use std::env;

pub struct CoinbaseClient {
    client: RestClient,
}

impl CoinbaseClient {
    /// Creates a new CoinbaseClient.
    ///
    /// It initializes the connection to the Coinbase Advanced Trade API
    /// using credentials from environment variables.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Retrieve API Key and Secret from environment variables
        let _api_key = env::var("COINBASE_API_KEY")
            .expect("COINBASE_API_KEY must be set in .env file or environment.");
        let _api_secret = env::var("COINBASE_API_SECRET")
            .expect("COINBASE_API_SECRET must be set in .env file or environment.");

        // Build the REST client, wiring API credentials from the environment
        // Try common builder methods to set the API key/secret on the RestClientBuilder.
        // If the cbadv crate uses different method names, the build step will reveal them and
        // we will iterate accordingly.
        let client: RestClient = RestClientBuilder::new()
            .with_authentication(&_api_key, &_api_secret)
            .build()?;

        Ok(Self { client })
    }

    /// Pings the Coinbase server to test the API connection.
    pub async fn test_connection(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.client.public.time().await {
            Ok(server_time) => {
                println!("Successfully connected to Coinbase!");
                println!("Server Time (ISO): {}", server_time.iso);
                Ok(())
            },
            Err(e) => {
                eprintln!("Error during test API call: {}", e);
                Err(Box::new(e) as Box<dyn std::error::Error>)
            }
        }
    }
}
