import requests
import json
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('employee_scraper.log'),
        logging.StreamHandler()
    ]
)

class EmployeeScraper:
    def __init__(self, api_url: str, max_retries: int = 3, timeout: int = 30):
        self.api_url = api_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def fetch_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch employee data from the API with retry logic and error handling.
        
        Returns:
            Dict containing the API response or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting to fetch data (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.get(
                    self.api_url,
                    timeout=self.timeout,
                    headers={'User-Agent': 'Employee-Scraper/1.0'}
                )
                
                if response.status_code == 200:
                    self.logger.info("Successfully fetched data from API")
                    return response.json()
                else:
                    self.logger.error(f"API returned status code {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                self.logger.error(f"Timeout error on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError:
                self.logger.error(f"Connection error on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {str(e)}")
                return None
            
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        self.logger.error("Failed to fetch data after all retry attempts")
        return None
    
    def validate_data_structure(self, data: Dict[str, Any]) -> bool:
        """
        Validate that the data contains the expected structure.
        
        Args:
            data: The JSON data from the API
            
        Returns:
            bool: True if data structure is valid
        """
        required_fields = [
            'id', 'first_name', 'last_name', 'email', 'job_title',
            'phone', 'gender', 'age', 'years_of_experience', 'salary', 'department'
        ]
        
        if not isinstance(data, list):
            self.logger.error("Data is not a list")
            return False
        
        if len(data) == 0:
            self.logger.error("Data list is empty")
            return False
        
        # Check first record for required fields
        first_record = data[0]
        missing_fields = [field for field in required_fields if field not in first_record]
        
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        
        self.logger.info("Data structure validation passed")
        return True
    
    def normalize_phone_number(self, phone: str) -> Optional[int]:
        """
        Normalize phone number and mark invalid ones.
        
        Args:
            phone: Phone number string
            
        Returns:
            int: Normalized phone number or None if invalid
        """
        if pd.isna(phone) or 'x' in str(phone).lower():
            return None
        
        # Remove all non-digit characters
        phone_digits = re.sub(r'\D', '', str(phone))
        
        try:
            return int(phone_digits) if phone_digits else None
        except ValueError:
            return None
    
    def determine_designation(self, years_experience: int) -> str:
        """
        Determine designation based on years of experience.
        
        Args:
            years_experience: Number of years of experience
            
        Returns:
            str: Designation title
        """
        if years_experience < 3:
            return "System Engineer"
        elif 3 <= years_experience < 5:  
            return "Data Engineer"
        elif 5 <= years_experience <= 10:  
            return "Senior Data Engineer"
        else:
            return "Lead"
    
    def format_date(self, date_str: str) -> Optional[str]:
        """
        Format date to YYYY-MM-DD format.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            str: Formatted date string or None if invalid
        """
        if pd.isna(date_str):
            return None
        
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(str(date_str), fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        self.logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def normalize_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Normalize and clean the employee data.
        
        Args:
            data: List of employee records
            
        Returns:
            pd.DataFrame: Normalized employee data
        """
        df = pd.DataFrame(data)
        
        # Create full name column
        df['full_name'] = df['first_name'].astype(str) + ' ' + df['last_name'].astype(str)
        
        # Normalize phone numbers
        df['phone'] = df['phone'].apply(self.normalize_phone_number)
        
        # Create designation column
        df['designation'] = df['years_of_experience'].apply(self.determine_designation)
        
        # Format hire date if it exists
        if 'hire_date' in df.columns:
            df['hire_date'] = df['hire_date'].apply(self.format_date)
        
        # Ensure correct data types
        type_mapping = {
            'full_name': 'string',
            'email': 'string',
            'phone': 'Int64',  # Nullable integer
            'gender': 'string',
            'age': 'int64',
            'job_title': 'string',
            'years_of_experience': 'int64',
            'salary': 'int64',
            'department': 'string',
            'designation': 'string'
        }
        
        for column, dtype in type_mapping.items():
            if column in df.columns:
                try:
                    df[column] = df[column].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Could not convert {column} to {dtype}: {str(e)}")
        
        # Select and reorder columns
        final_columns = [
            'id', 'full_name', 'email', 'phone', 'gender', 'age',
            'job_title', 'years_of_experience', 'salary', 'department', 'designation'
        ]
        
        if 'hire_date' in df.columns:
            final_columns.append('hire_date')
        
        df = df[final_columns]
        
        self.logger.info(f"Data normalization completed. Shape: {df.shape}")
        return df
    
    def process_employees(self) -> Optional[pd.DataFrame]:
        """
        Main method to process employee data.
        
        Returns:
            pd.DataFrame: Processed employee data or None if failed
        """
        # Fetch data
        raw_data = self.fetch_data()
        if raw_data is None:
            return None
        
        # Validate structure
        if not self.validate_data_structure(raw_data):
            return None
        
        # Normalize data
        try:
            normalized_df = self.normalize_data(raw_data)
            self.logger.info("Employee data processing completed successfully")
            return normalized_df
        except Exception as e:
            self.logger.error(f"Error during data normalization: {str(e)}")
            return None

def main():
    """Main execution function"""
    api_url = "https://api.slingacademy.com/v1/sample-data/files/employees.json"
    
    scraper = EmployeeScraper(api_url)
    result_df = scraper.process_employees()
    
    if result_df is not None:
        print("Employee Data Processing Summary:")
        print(f"Total records processed: {len(result_df)}")
        print(f"Columns: {list(result_df.columns)}")
        print("\nFirst 5 records:")
        print(result_df.head())
        
        # Save to CSV
        output_file = "processed_employees.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        
        # Display data quality metrics
        print("\nData Quality Metrics:")
        print(f"Records with invalid phone numbers: {result_df['phone'].isna().sum()}")
        print(f"Designation distribution:")
        print(result_df['designation'].value_counts())
    else:
        print("Failed to process employee data")

if __name__ == "__main__":
    main()
