import pandas as pd
from src.configuration.mongo_db_connection import MongoDBClient

# File path
CSV_FILE_PATH = r"Notebook\data\clustered_data.csv"


COLLECTION_NAME = "customer_segmentation"

if __name__ == "__main__":
    # Read CSV
    df = pd.read_csv(CSV_FILE_PATH)

    print(f"CSV loaded with shape: {df.shape}")

    # Convert DataFrame to dict
    records = df.to_dict(orient="records")

    # MongoDB connection
    mongo_client = MongoDBClient()
    collection = mongo_client.database[COLLECTION_NAME]

    # Insert data
    result = collection.insert_many(records)

    print(f"Inserted {len(result.inserted_ids)} documents into MongoDB ✅")
