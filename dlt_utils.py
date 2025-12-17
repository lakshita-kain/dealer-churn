from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame
from typing import List, Optional
import uuid
from datetime import datetime
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DLTReader:
    """
    A utility class to read Delta Live Tables from a specified
    Databricks catalog and schema, following best practices.
    """
    def __init__(self, catalog: str = "provisioned-tableau-data", schema: str = "tableau_delta_tables"):
        """
        Initializes the DLTReader and establishes the Databricks Spark session.

        Args:
            catalog (str): The name of the data catalog to use.
            schema (str): The name of the schema (database) to use.
        """
        self.catalog = catalog
        self.schema = schema
        self.spark = self._create_spark_session()
        self._set_session_context()

    def _create_spark_session(self) -> DatabricksSession:
        """
        Creates and returns a configured Databricks Spark session.
        This is separated for clarity and potential future extensions.
        """
        print("⚡ Initializing Databricks Spark session...")
        session = DatabricksSession.builder.getOrCreate()
        print("✅ Session initialized successfully.")
        return session

    def _set_session_context(self):
        """
        Sets the default catalog and schema for the current Spark session
        to simplify queries.
        """
        print(f"📁 Setting catalog to: `{self.catalog}`")
        self.spark.sql(f"USE CATALOG `{self.catalog}`")
        print(f"📄 Setting schema to: `{self.schema}`")
        self.spark.sql(f"USE SCHEMA `{self.schema}`")

    def read_table(self, table_name: str, columns: Optional[List[str]] = None, limit: Optional[int] = None) -> DataFrame:
        """
        Reads a specific table from the configured schema into a Spark DataFrame.

        Args:
            table_name (str): The name of the table to read (e.g., 'product_master').
            columns (Optional[List[str]]): A list of specific column names to select.
                                           If None, all columns ('*') are selected.
            limit (Optional[int]): If provided, limits the number of rows returned.

        Returns:
            DataFrame: A Spark DataFrame containing the requested table's data.
        """
        print(f"📖 Reading table: `{table_name}`...")
        query = f"SELECT {', '.join(columns)} FROM `{self.catalog}`.`{self.schema}`.`{table_name}`" if columns else f"SELECT * FROM `{self.catalog}`.`{self.schema}`.`{table_name}`"
        
        if limit is not None and isinstance(limit, int) and limit > 0:
            query += f" LIMIT {limit}"
        
        try:
            df = self.spark.sql(query)
            print(f"✅ Successfully read data from `{table_name}`.")
            return df
        except Exception as e:
            print(f"❌ Error reading table `{table_name}`: {e}")
            # Return an empty DataFrame with the same schema to prevent downstream errors
            return self.spark.createDataFrame([], self.spark.sql(f"SELECT * FROM `{table_name}` LIMIT 0").schema)

    def list_tables(self) -> List[str]:
        """
        Lists all tables available in the currently configured schema.

        Returns:
            List[str]: A list of table names.
        """
        print(f"📋 Listing all tables in `{self.catalog}`.`{self.schema}`...")
        try:
            tables_df = self.spark.sql("SHOW TABLES")
            table_names = [row.tableName for row in tables_df.collect()]
            print(f"✅ Found {len(table_names)} tables.")
            return table_names
        except Exception as e:
            print(f"❌ Error listing tables: {e}")
            return []


class DLTWriter:
    """
    A utility class to write DataFrames to Delta Live Tables in the data_science schema,
    following best practices for data science workflows.
    """
    def __init__(self, catalog: str = "provisioned-tableau-data", schema: str = "data_science"):
        """
        Initializes the DLTWriter and establishes the Databricks Spark session.

        Args:
            catalog (str): The name of the data catalog to use.
            schema (str): The name of the schema (database) to use for data science outputs.
        """
        self.catalog = catalog
        self.schema = schema
        self.spark = self._create_spark_session()
        self._set_session_context()

    def _create_spark_session(self) -> DatabricksSession:
        """
        Creates and returns a configured Databricks Spark session.
        This is separated for clarity and potential future extensions.
        """
        print("⚡ Initializing Databricks Spark session for writing...")
        session = DatabricksSession.builder.getOrCreate()
        print("✅ Session initialized successfully.")
        return session

    def _set_session_context(self):
        """
        Sets the default catalog and schema for the current Spark session
        to simplify operations.
        """
        print(f"📁 Setting catalog to: `{self.catalog}`")
        # self.spark.sql(f"USE CATALOG `{self.catalog}`")
        # TODO: Comment this out for now
        # print(f"📄 Setting schema to: `{self.schema}`")
        # self.spark.sql(f"USE SCHEMA `{self.schema}`")

    def write_table(self, 
                   df: DataFrame, 
                   table_name: str, 
                   format: str = "delta",
                   mode: str = "overwrite",
                   partition_by: Optional[List[str]] = None,
                   merge_schema: bool = True) -> bool:
        """
        Writes a DataFrame to a Delta table in the data_science schema.

        Args:
            df (DataFrame): The Spark DataFrame to write.
            table_name (str): The name of the table to create/update.
            mode (str): Write mode - 'overwrite', 'append', 'errorIfExists', 'ignore'.
            partition_by (Optional[List[str]]): List of columns to partition by.
            merge_schema (bool): Whether to merge schema if table exists.

        Returns:
            bool: True if successful, False otherwise.
        """
        print(f"📝 Writing DataFrame to table: `{table_name}`...")
        print(f"   Mode: {mode}")
        print(f"   Partition by: {partition_by if partition_by else 'None'}")
        print(f"   Merge schema: {merge_schema}")
        
        try:
            if isinstance(df, pd.DataFrame):
                print("🔄 Converting Pandas DataFrame to Spark DataFrame...")
                df = self.spark.createDataFrame(df)

            # Build the write operation
            writer = df.write.format(format).mode(mode)
            
            if partition_by:
                writer = writer.partitionBy(*partition_by)
            
            if merge_schema:
                writer = writer.option("mergeSchema", "true")
            
            # Write the table
            # TODO: Prepend the schema name to the table name
            print(f"📄 Setting schema to: `{self.schema}`")
            writer.saveAsTable(f"`{self.catalog}`.`{self.schema}`.`{table_name}`")
            
            print(f"✅ Successfully wrote data to `{table_name}`.")
            return True
            
        except Exception as e:
            print(f"❌ Error writing to table `{table_name}`: {e}")
            return False

    def write_table_with_timestamp(self, 
                                  df: DataFrame, 
                                  base_table_name: str,
                                  mode: str = "overwrite",
                                  partition_by: Optional[List[str]] = None,
                                  merge_schema: bool = True) -> str:
        """
        Writes a DataFrame to a timestamped table for versioning and tracking.

        Args:
            df (DataFrame): The Spark DataFrame to write.
            base_table_name (str): The base name for the table.
            mode (str): Write mode - 'overwrite', 'append', 'errorIfExists', 'ignore'.
            partition_by (Optional[List[str]]): List of columns to partition by.
            merge_schema (bool): Whether to merge schema if table exists.

        Returns:
            str: The actual table name that was created.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_name = f"{base_table_name}_{timestamp}"
        
        success = self.write_table(df, table_name, mode, partition_by, merge_schema)
        if success:
            return table_name
        else:
            return ""

    def write_table_with_uuid(self, 
                             df: DataFrame, 
                             base_table_name: str,
                             mode: str = "overwrite",
                             partition_by: Optional[List[str]] = None,
                             merge_schema: bool = True) -> str:
        """
        Writes a DataFrame to a UUID-based table for unique identification.

        Args:
            df (DataFrame): The Spark DataFrame to write.
            base_table_name (str): The base name for the table.
            mode (str): Write mode - 'overwrite', 'append', 'errorIfExists', 'ignore'.
            partition_by (Optional[List[str]]): List of columns to partition by.
            merge_schema (bool): Whether to merge schema if table exists.

        Returns:
            str: The actual table name that was created.
        """
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        table_name = f"{base_table_name}_{unique_id}"
        
        success = self.write_table(df, table_name, mode, partition_by, merge_schema)
        if success:
            return table_name
        else:
            return ""

    def write_clustering_features(self, 
                                 df: DataFrame, 
                                 model_name: str = "dealer_clustering",
                                 version: str = "v1") -> str:
        """
        Specialized method for writing clustering features with proper naming and metadata.

        Args:
            df (DataFrame): The clustering features DataFrame.
            model_name (str): Name of the clustering model.
            version (str): Version of the model.

        Returns:
            str: The table name that was created.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_name = f"{model_name}_features_{version}_{timestamp}"
        
        print(f"🎯 Writing clustering features to: `{table_name}`")
        print(f"   Model: {model_name}")
        print(f"   Version: {version}")
        print(f"   Timestamp: {timestamp}")
        
        success = self.write_table(df, table_name, mode="overwrite")
        if success:
            print(f"✅ Clustering features saved successfully!")
            return table_name
        else:
            print(f"❌ Failed to save clustering features!")
            return ""

    def list_tables(self) -> List[str]:
        """
        Lists all tables available in the data_science schema.

        Returns:
            List[str]: A list of table names.
        """
        print(f"📋 Listing all tables in `{self.catalog}`.`{self.schema}`...")
        try:
            tables_df = self.spark.sql("SHOW TABLES")
            table_names = [row.tableName for row in tables_df.collect()]
            print(f"✅ Found {len(table_names)} tables.")
            return table_names
        except Exception as e:
            print(f"❌ Error listing tables: {e}")
            return []

    def drop_table(self, table_name: str) -> bool:
        """
        Drops a table from the data_science schema.

        Args:
            table_name (str): The name of the table to drop.

        Returns:
            bool: True if successful, False otherwise.
        """
        print(f"🗑️ Dropping table: `{table_name}`...")
        try:
            self.spark.sql(f"DROP TABLE IF EXISTS `{table_name}`")
            print(f"✅ Successfully dropped table `{table_name}`.")
            return True
        except Exception as e:
            print(f"❌ Error dropping table `{table_name}`: {e}")
            return False

    def table_exists(self, table_name: str) -> bool:
        """
        Checks if a table exists in the data_science schema.

        Args:
            table_name (str): The name of the table to check.

        Returns:
            bool: True if table exists, False otherwise.
        """
        try:
            tables_df = self.spark.sql("SHOW TABLES")
            existing_tables = [row.tableName for row in tables_df.collect()]
            return table_name in existing_tables
        except Exception as e:
            print(f"❌ Error checking if table exists: {e}")
            return False


# Example of how to use the DLTReader and DLTWriter classes.
# This block runs only when you execute `python dlt_utils.py` in your terminal.
if __name__ == "__main__":
    print("🚀 Running DLTReader and DLTWriter Utility Example")
    print("="*50)

    try:
        # 1. Initialize the reader and writer
        dlt_reader = DLTReader()
        dlt_writer = DLTWriter()

        # 2. List all available tables in tableau_delta_tables
        print("\n📊 Tables in tableau_delta_tables schema:")
        available_tables = dlt_reader.list_tables()
        if available_tables:
            for table in available_tables[:5]:  # Show first 5
                print(f"  - {table}")
            if len(available_tables) > 5:
                print(f"  ... and {len(available_tables) - 5} more.")

        # 3. List all available tables in data_science
        print("\n📊 Tables in data_science schema:")
        ds_tables = dlt_writer.list_tables()
        if ds_tables:
            for table in ds_tables[:5]:  # Show first 5
                print(f"  - {table}")
            if len(ds_tables) > 5:
                print(f"  ... and {len(ds_tables) - 5} more.")
        else:
            print("  (No tables found - schema might be empty)")

        # 4. Example: Read a table and write it to data_science
        if 'product_master' in available_tables:
            print("\n" + "="*50)
            print("🔄 Example: Reading from tableau_delta_tables and writing to data_science")
            
            # Read sample data
            product_df = dlt_reader.read_table('product_master', limit=100)
            
            if product_df.count() > 0:
                # Write to data_science with timestamp
                table_name = dlt_writer.write_table_with_timestamp(
                    product_df, 
                    "product_master_sample"
                )
                
                if table_name:
                    print(f"✅ Successfully wrote sample data to: {table_name}")
                    
                    # Verify the table was created
                    if dlt_writer.table_exists(table_name):
                        print(f"✅ Verified table exists: {table_name}")
                    else:
                        print(f"❌ Table not found: {table_name}")
                else:
                    print("❌ Failed to write table")
            else:
                print("❌ No data to write")
        else:
            print("\nCould not find 'product_master' table to run example.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the example run: {e}")
    
    print("\n✅ Example finished.") 