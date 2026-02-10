import pandas as pd
from dotenv import load_dotenv
import os
import json
import logging
from typing import Dict, List
from pydantic import BaseModel

# Import Euri API Client and DataSourceType
from euri_client import MultiKeyEuriClient, DataSourceType

# Load API key from environment
load_dotenv()

# Initialize Euri API Client
euri_client = MultiKeyEuriClient()

class CleaningState(BaseModel):
    """State schema defining input and output for the AI agent."""
    input_text: str
    structured_response: str = ""

class AIAgent:
    def __init__(self):
        """
        Initialize AI Agent with Multi-Key Euri Client
        """
        self.client = euri_client
        self.logger = logging.getLogger(__name__)
        print("✅ AI Agent initialized with Multi-Key Euri Client")

    def process_data(self, df, source_type: DataSourceType = DataSourceType.DEFAULT, batch_size: int = 20):
        """
        Processes data in batches using Euri API to avoid token limits.
        
        Args:
            df: Pandas DataFrame to clean
            source_type: The source of the data for routing (DB, Upload, etc.)
            batch_size: Number of rows to process at once
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_rows = []
        
        # Determine model for logging
        config = self.client.get_config_for_source(source_type)
        print(f"🤖 AI Processing via {source_type.value} route using model: {config.model}")

        for i in range(0, len(df), batch_size):
            df_batch = df.iloc[i:i + batch_size]

            # Use to_csv for better representation to the AI
            data_csv = df_batch.to_csv(index=False)

            prompt = f"""
You are an AI Data Cleaning Expert. Analyze this dataset and clean it.

CRITICAL INSTRUCTIONS:
1. PRESERVE ALL COLUMNS: You must return ALL columns present in the input. Do not drop columns unless they are 100% empty or metadata noise.
2. HUMAN-UNDERSTANDABLE CONTENT: If you encounter placeholder text (like "Lorem Ipsum" or "Latin dummy text"), do not just return it as is. Instead, provide a brief, human-understandable English summary or paraphrase of the intended meaning if possible, or replace it with clear, descriptive English text that fits the context.
3. OUTPUT FORMAT: Return ONLY the cleaned data in valid CSV format. Do not include any explanations, markdown blocks, or preamble.

Input Dataset (CSV):
{data_csv}

Tasks:
1. Handle missing values (impute based on context).
2. Remove duplicates.
3. Fix inconsistent formatting and typos.
4. Ensure text data is grammatically correct and in plain English.
5. Standardize types and remove obvious outliers.
"""

            try:
                # Get response from Euri API using source-aware routing
                response = self.client.get_text_completion(
                    prompt=prompt,
                    source_type=source_type,
                    temperature=0.3
                )
                
                # Parse the CSV response
                from io import StringIO
                cleaned_csv_text = response.strip()
                # Remove markdown code blocks if AI included them despite instructions
                if "```csv" in cleaned_csv_text:
                    cleaned_csv_text = cleaned_csv_text.split("```csv")[1].split("```")[0].strip()
                elif "```" in cleaned_csv_text:
                    cleaned_csv_text = cleaned_csv_text.split("```")[1].split("```")[0].strip()
                
                cleaned_batch = pd.read_csv(StringIO(cleaned_csv_text))
                
                # Ensure batch has same columns as original
                for col in df.columns:
                    if col not in cleaned_batch.columns:
                        cleaned_batch[col] = df_batch[col].values
                
                cleaned_rows.append(cleaned_batch)
                
            except Exception as e:
                print(f"⚠️ Warning: Batch {i//batch_size + 1} processing failed: {str(e)}")
                # If AI processing fails, return original batch
                cleaned_rows.append(df_batch)

        # Combine all cleaned batches
        if cleaned_rows:
            result_df = pd.concat(cleaned_rows, ignore_index=True)
            return result_df
        else:
            return df

    def analyze_data_quality(self, df) -> Dict:
        """
        Analyze data quality and provide insights using AI
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics and recommendations
        """
        quality_metrics = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict()
        }

        prompt = f"""
As a Data Scientist, analyze the quality of this dataset summary:
{json.dumps(quality_metrics, indent=2)}

Provide:
1. Assessment of data health
2. Specific recommendations for improvement
3. Potential use cases for this data
"""

        try:
            insight = self.client.get_text_completion(
                prompt=prompt,
                source_type=DataSourceType.DEFAULT
            )
            quality_metrics["ai_insights"] = insight
        except:
            quality_metrics["ai_insights"] = "AI Analysis unavailable"

        return quality_metrics
