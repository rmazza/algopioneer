pub mod websocket;
use cbadv::{RestClient, RestClientBuilder};
use std::env;
use crate::sandbox;
use cbadv::models::product::{Candle, ProductCandleQuery};
use cbadv::time::Granularity;
use chrono::{DateTime, Utc};

pub enum AppEnv {
    Live,
    Sandbox,
    Paper,
}

pub struct CoinbaseClient {
    client: RestClient,
    mode: AppEnv,
}

impl CoinbaseClient {
    /// Creates a new CoinbaseClient.
    ///
    /// It initializes the connection to the Coinbase Advanced Trade API
    /// using credentials from environment variables.
    pub fn new(env: AppEnv) -> Result<Self, Box<dyn std::error::Error>> {
        // Retrieve API Key and Secret from environment variables
        let _api_key = env::var("COINBASE_API_KEY")
            .expect("COINBASE_API_KEY must be set in .env file or environment.");
        let _api_secret = env::var("COINBASE_API_SECRET")
            .expect("COINBASE_API_SECRET must be set in .env file or environment.");

        // Build the REST client, wiring API credentials from the environment
        let client: RestClient = RestClientBuilder::new()
            .with_authentication(&_api_key, &_api_secret)
            .build()?;

        Ok(Self { client, mode: env })
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

    /// Places an order.
    pub async fn place_order(&self, product_id: &str, side: &str, size: f64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self.mode {
            AppEnv::Live => {
                println!("-- Live Mode: Placing order for {} {} of {} --", side, size, product_id);
                // Here you would add the actual call to the Coinbase API to place an order
                // For now, we just print a message.
                Ok(())
            },
            AppEnv::Sandbox => {
                let trade_details = format!("{},{},{}", product_id, side, size);
                sandbox::save_trade(&trade_details)
            },
            AppEnv::Paper => {
                let msg = format!("-- PAPER TRADE: {} {} of {} --", side, size, product_id);
                println!("{}", msg);
                
                // Log to CSV
                use std::fs::OpenOptions;
                use std::io::Write;
                
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("paper_trades.csv")?;
                
                writeln!(file, "{},{},{},{}", Utc::now(), product_id, side, size)?;
                Ok(())
            }
        }
    }

    /// Gets product candles (candlestick data).
    pub async fn get_product_candles(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
        let start_timestamp = start.timestamp() as u64;
        let end_timestamp = end.timestamp() as u64;

        let query = ProductCandleQuery {
            start: start_timestamp,
            end: end_timestamp,
            granularity: Granularity::OneMinute,
            limit: 300,
        };

        let candles = self
            .client
            .product
            .candles(product_id, &query)
            .await?;
        Ok(candles)
    }
}
