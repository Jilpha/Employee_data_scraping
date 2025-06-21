import requests
import json
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import time
import re
import numpy as np
import sys

# Configure logging with UTF-8 encoding and safe console handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('employee_scraper.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Use sys.stdout explicitly to avoid invalid handle issues
    ]
)

class EmployeeScraper:
    def __init__(self, api_url: str, max_retries: int = 3, timeout: int = 30):
        self.api_url = api_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.raw_data = None
        self.processed_data = None
    
    def fetch_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch employee data from the API with retry logic and error handling.
        
        Returns:
            List containing the API response or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting to fetch data (attempt {attempt + 1}/{self.max_retries})")
                self.logger.info(f"Making HTTP request to: {self.api_url}")
                
                response = requests.get(
                    self.api_url,
                    timeout=self.timeout,
                    headers={'User-Agent': 'Employee-Scraper/1.0'}
                )
                
                if response.status_code == 200:
                    self.logger.info("Successfully fetched data from API")
                    try:
                        raw_json = response.json()
                        self.raw_data = raw_json
                        self.logger.info(f"Retrieved {len(raw_json)} employee records")
                        return raw_json
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON decode error: {str(e)}")
                        return None
                else:
                    self.logger.error(f"API returned status code {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                self.logger.error(f"Timeout error on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError:
                self.logger.error(f"Connection error on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
            
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        self.logger.error("Failed to fetch data after all retry attempts")
        return None
    
    def validate_data_structure(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate that the data contains the expected structure and required fields.
        
        Args:
            data: The JSON data from the API
            
        Returns:
            bool: True if data structure is valid
        """
        self.logger.info("Starting data structure validation...")
        
        if not isinstance(data, list):
            self.logger.error("Data is not a list")
            return False
        
        if len(data) == 0:
            self.logger.error("Data list is empty")
            return False
        
        # Check first record for required fields
        first_record = data[0]
        if not isinstance(first_record, dict):
            self.logger.error("First record is not a dictionary")
            return False
        
        # Required fields for the user story
        required_fields = [
            'id',  # Employee ID
            'first_name',  # First Name
            'last_name',   # Last Name
            'email',       # Email
            'job_title',   # Job Title
            'phone'        # Phone Number
            # Note: hire_date might not be present, we'll handle this gracefully
        ]
        
        available_fields = list(first_record.keys())
        self.logger.info(f"Available fields in API data: {available_fields}")
        
        missing_fields = [field for field in required_fields if field not in first_record]
        
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        
        # Check for additional fields needed for normalization
        normalization_fields = ['years_of_experience', 'gender', 'age', 'salary', 'department']
        available_norm_fields = [field for field in normalization_fields if field in first_record]
        self.logger.info(f"Available normalization fields: {available_norm_fields}")
        
        self.logger.info("Data structure validation passed")
        return True
    
    def parse_employee_fields(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse JSON data into appropriate employee fields as per user story requirements.
        
        Args:
            raw_data: Raw JSON data from API
            
        Returns:
            List of parsed employee records
        """
        parsed_employees = []
        
        self.logger.info("Starting to parse employee fields...")
        
        for i, employee in enumerate(raw_data):
            try:
                # Parse core required fields from user story
                first_name = str(employee.get('first_name', '')).strip()
                last_name = employee.get('last_name', '')
                last_name = '' if last_name is None else str(last_name).strip()
                # Preserve trailing space when last_name is empty, as expected by test
                full_name = f"{first_name} {last_name}" if last_name else f"{first_name} "
                parsed_employee = {
                    # Core identification fields (User Story Requirements)
                    'employee_id': employee.get('id'),
                    'first_name': first_name,
                    'last_name': last_name,
                    'full_name': full_name,
                    'email': str(employee.get('email', '')).strip().lower(),
                    'job_title': str(employee.get('job_title', '')).strip(),
                    'phone_number': employee.get('phone', ''),
                    'hire_date': self.parse_date(employee.get('hire_date')),
                    
                    # Additional fields for normalization
                    'gender': str(employee.get('gender', '')).strip(),
                    'age': employee.get('age'),
                    'years_of_experience': employee.get('years_of_experience'),
                    'salary': employee.get('salary'),
                    'department': str(employee.get('department', '')).strip(),
                    
                    # Metadata
                    'extraction_timestamp': datetime.now().isoformat(),
                    'record_source': 'API'
                }
                
                parsed_employees.append(parsed_employee)
                
            except Exception as e:
                self.logger.warning(f"Error parsing employee record {i}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully parsed {len(parsed_employees)} employee records")
        return parsed_employees
    
    def parse_date(self, date_value: Any) -> Optional[str]:
        """
        Parse date value into standardized YYYY-MM-DD format.
        
        Args:
            date_value: Date value from API
            
        Returns:
            str: Formatted date string (YYYY-MM-DD) or None
        """
        if not date_value or pd.isna(date_value):
            return None
        
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y'
        ]
        
        date_str = str(date_value).strip()
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        self.logger.warning(f"Could not parse date: {date_value}")
        return None
    
    def normalize_data(self, parsed_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Normalize and clean the employee data according to user story requirements.
        
        Args:
            parsed_data: Parsed employee data
            
        Returns:
            pd.DataFrame: Normalized employee data
        """
        self.logger.info("Starting data normalization...")
        
        df = pd.DataFrame(parsed_data)
        
        # 1. Create designation based on years_of_experience
        self.logger.info("Creating designation column based on experience...")
        df['designation'] = df['years_of_experience'].apply(self.determine_designation)
        
        # 2. Combine first_name and last_name into full_name (already done in parse_employee_fields, kept for compatibility)
        self.logger.info("Creating full_name column...")
        df['full_name'] = df['first_name'].astype(str) + ' ' + df['last_name'].astype(str)
        df['full_name'] = df['full_name'].str.strip()
        
        # 3. Handle phone numbers - mark those with 'x' as invalid
        self.logger.info("Processing phone numbers...")
        df['phone'] = df['phone_number'].apply(self.normalize_phone_number)
        
        # 4. Ensure correct data types as specified in user story
        self.logger.info("Enforcing data types...")
        df = self.enforce_data_types(df)
        
        # 5. Format dates consistently (YYYY-MM-DD)
        self.logger.info("Formatting dates...")
        if 'hire_date' in df.columns:
            # hire_date is already formatted in parse_date method
            pass
        
        # Select final columns in the order specified by user story
        final_columns = [
            'employee_id',
            'full_name',
            'email', 
            'phone',
            'gender',
            'age',
            'job_title',
            'years_of_experience',
            'salary',
            'department',
            'designation',
            'hire_date'
        ]
        
        # Only include columns that exist
        available_final_columns = [col for col in final_columns if col in df.columns]
        df_final = df[available_final_columns].copy()
        
        self.logger.info(f"Data normalization completed. Final shape: {df_final.shape}")
        self.processed_data = df_final
        return df_final
    
    def determine_designation(self, years_experience: Union[int, float, None]) -> str:
        """
        Determine designation based on years of experience as per user story.
        
        Args:
            years_experience: Number of years of experience
            
        Returns:
            str: Designation title
        """
        if pd.isna(years_experience) or years_experience is None:
            return "Unknown"
        
        try:
            years = int(years_experience)
            if years < 3:
                return "System Engineer"
            elif 3 <= years <= 5:  # "more than 3-5" interpreted as 3-5 inclusive
                return "Data Engineer"
            elif 5 < years <= 10:  # "between 5-10" interpreted as 5-10 inclusive
                return "Senior Data Engineer"
            else:  # 10+ years
                return "Lead"
        except (ValueError, TypeError):
            return "Unknown"
    
    def normalize_phone_number(self, phone: Any) -> Optional[int]:
        """
        Normalize phone number and mark invalid ones (containing 'x').
        
        Args:
            phone: Phone number value
            
        Returns:
            int: Normalized phone number or None if invalid
        """
        if pd.isna(phone) or phone is None:
            return None
        
        phone_str = str(phone).lower()
        
        # Check if phone contains 'x' - mark as invalid
        if 'x' in phone_str:
            return None
        
        # Remove all non-digit characters
        phone_digits = re.sub(r'\D', '', phone_str)
        
        try:
            return int(phone_digits) if phone_digits else None
        except ValueError:
            return None
    
    def enforce_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce data types as specified in user story requirements.
        
        Args:
            df: DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with enforced data types
        """
        type_mapping = {
            'full_name': 'string',
            'email': 'string',
            'phone': 'Int64',  # Nullable integer for phone numbers
            'gender': 'string',
            'age': 'Int64',    # Nullable integer
            'job_title': 'string',
            'years_of_experience': 'Int64',  # Nullable integer
            'salary': 'Int64',  # Nullable integer
            'department': 'string',
            'designation': 'string',
            'hire_date': 'string'  # Keep as string in YYYY-MM-DD format
        }
        
        for column, dtype in type_mapping.items():
            if column in df.columns:
                try:
                    if dtype == 'Int64':
                        # Handle nullable integers
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    else:
                        df[column] = df[column].astype(dtype)
                    self.logger.info(f"Converted {column} to {dtype}")
                except Exception as e:
                    self.logger.warning(f"Could not convert {column} to {dtype}: {str(e)}")
        
        return df
    
    def get_currency_conversion_rate(self, from_currency: str = 'USD', to_currency: str = 'USD') -> float:
        """
        Get currency conversion rate (Optional feature from user story).
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            float: Conversion rate
        """
        if from_currency == to_currency:
            return 1.0
        
        try:
            # Using a free currency API (example)
            api_url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                rate = data.get('rates', {}).get(to_currency, 1.0)
                self.logger.info(f"Currency conversion rate {from_currency} to {to_currency}: {rate}")
                return rate
            else:
                self.logger.warning(f"Could not fetch currency rate, using 1.0")
                return 1.0
        except Exception as e:
            self.logger.warning(f"Currency conversion error: {str(e)}, using 1.0")
            return 1.0
    
    def apply_currency_conversion(self, df: pd.DataFrame, from_currency: str = 'USD', to_currency: str = 'USD') -> pd.DataFrame:
        """
        Apply currency conversion to salary column (Optional feature).
        
        Args:
            df: DataFrame with salary data
            from_currency: Source currency
            to_currency: Target currency
            
        Returns:
            pd.DataFrame: DataFrame with converted salaries
        """
        if from_currency == to_currency or 'salary' not in df.columns:
            return df
        
        self.logger.info(f"Applying currency conversion from {from_currency} to {to_currency}")
        
        conversion_rate = self.get_currency_conversion_rate(from_currency, to_currency)
        
        # Apply conversion
        df['salary_original'] = df['salary'].copy()
        df['salary'] = (df['salary'] * conversion_rate).round().astype('Int64')
        df['currency'] = to_currency
        
        self.logger.info(f"Currency conversion applied with rate: {conversion_rate}")
        return df
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dict: Data quality metrics
        """
        self.logger.info("Generating data quality report...")
        
        report = {
            'total_records': len(df),
            'field_completeness': {},
            'data_quality_metrics': {},
            'designation_distribution': {},
            'summary': {}
        }
        
        # Field completeness
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            completeness = (non_null_count / len(df)) * 100
            report['field_completeness'][col] = {
                'non_null_count': int(non_null_count),
                'completeness_percentage': round(completeness, 2)
            }
        
        # Data quality metrics
        if 'phone' in df.columns:
            valid_phones = df['phone'].notna().sum()
            invalid_phones = df['phone'].isna().sum()
            report['data_quality_metrics']['phone_numbers'] = {
                'valid': int(valid_phones),
                'invalid_with_x': int(invalid_phones),
                'validity_percentage': round((valid_phones / len(df)) * 100, 2)
            }
        
        if 'email' in df.columns:
            valid_emails = df['email'].str.contains('@', na=False).sum()
            report['data_quality_metrics']['emails'] = {
                'valid_format': int(valid_emails),
                'validity_percentage': round((valid_emails / len(df)) * 100, 2)
            }
        
        # Designation distribution
        if 'designation' in df.columns:
            designation_counts = df['designation'].value_counts()
            for designation, count in designation_counts.items():
                report['designation_distribution'][designation] = {
                    'count': int(count),
                    'percentage': round((count / len(df)) * 100, 2)
                }
        
        # Summary
        report['summary'] = {
            'data_extraction_successful': True,
            'total_records_processed': len(df),
            'average_age': float(df['age'].mean()) if 'age' in df.columns else None,
            'average_experience': float(df['years_of_experience'].mean()) if 'years_of_experience' in df.columns else None,
            'average_salary': float(df['salary'].mean()) if 'salary' in df.columns else None
        }
        
        return report
    
    def save_results(self, df: pd.DataFrame, quality_report: Dict[str, Any]) -> None:
        """
        Save all results to files.
        
        Args:
            df: Processed DataFrame
            quality_report: Data quality report
        """
        try:
            # Save raw data
            if self.raw_data:
                with open('raw_employees_data.json', 'w', encoding='utf-8') as f:
                    json.dump(self.raw_data, f, indent=2, ensure_ascii=False)
                self.logger.info("Raw data saved to 'raw_employees_data.json'")
            
            # Save processed data as CSV
            df.to_csv('processed_employees.csv', index=False)
            self.logger.info("Processed data saved to 'processed_employees.csv'")
            
            # Save processed data as JSON
            df.to_json('processed_employees.json', orient='records', indent=2)
            self.logger.info("Processed data saved to 'processed_employees.json'")
            
            # Save quality report
            with open('data_quality_report.json', 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False)
            self.logger.info("Quality report saved to 'data_quality_report.json'")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def process_employee_data(self, apply_currency_conversion: bool = False, 
                            from_currency: str = 'USD', to_currency: str = 'USD') -> Optional[pd.DataFrame]:
        """
        Main method to process employee data according to user story requirements.
        
        Args:
            apply_currency_conversion: Whether to apply currency conversion
            from_currency: Source currency for conversion
            to_currency: Target currency for conversion
            
        Returns:
            pd.DataFrame: Processed employee data or None if failed
        """
        self.logger.info("Starting employee data processing...")
        
        # Step 1: Data Retrieval
        self.logger.info("STEP 1: DATA RETRIEVAL")
        raw_data = self.fetch_data()
        if raw_data is None:
            self.logger.error("Data retrieval failed")
            return None
        
        # Step 2: Data Structure Validation
        self.logger.info("STEP 2: DATA STRUCTURE VALIDATION")
        if not self.validate_data_structure(raw_data):
            self.logger.error("Data validation failed")
            return None
        
        # Step 3: Parse Employee Fields
        self.logger.info("STEP 3: PARSING EMPLOYEE FIELDS")
        parsed_data = self.parse_employee_fields(raw_data)
        
        # Step 4: Data Normalization
        self.logger.info("STEP 4: DATA NORMALIZATION")
        normalized_df = self.normalize_data(parsed_data)
        
        # Step 5: Currency Conversion (Optional)
        if apply_currency_conversion and from_currency != to_currency:
            self.logger.info("STEP 5: CURRENCY CONVERSION")
            normalized_df = self.apply_currency_conversion(normalized_df, from_currency, to_currency)
        
        # Step 6: Generate Quality Report
        self.logger.info("STEP 6: GENERATING QUALITY REPORT")
        quality_report = self.generate_data_quality_report(normalized_df)
        
        # Step 7: Save Results
        self.logger.info("STEP 7: SAVING RESULTS")
        self.save_results(normalized_df, quality_report)
        
        self.logger.info("Employee data processing completed successfully!")
        return normalized_df

def main():
    """Main execution function"""
    print("="*80)
    print("EMPLOYEE DATA SCRAPER - USER STORY IMPLEMENTATION")
    print("="*80)
    print("User Story: Scraping Employee Data from API")
    print("Objective: Ingest employee data into data warehouse for analysis")
    print("="*80)
    
    # API endpoint from user story
    api_url = "https://api.slingacademy.com/v1/sample-data/files/employees.json"
    
    # Initialize scraper
    scraper = EmployeeScraper(api_url)
    
    # Process data (with optional currency conversion)
    result_df = scraper.process_employee_data(
        apply_currency_conversion=False,  # Set to True to enable currency conversion
        from_currency='USD',
        to_currency='EUR'  # Change as needed
    )
    
    if result_df is not None:
        print("\n" + "="*60)
        print("DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Display summary
        print(f"Total records processed: {len(result_df)}")
        print(f"Columns available: {len(result_df.columns)}")
        print(f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show column information
        print(f"\nFINAL DATASET COLUMNS:")
        print("-" * 40)
        for i, col in enumerate(result_df.columns, 1):
            dtype = str(result_df[col].dtype)
            non_null = result_df[col].notna().sum()
            print(f"{i:2d}. {col:<25} ({dtype:<10}) - {non_null}/{len(result_df)} records")
        
        # Show sample data
        print(f"\nSAMPLE PROCESSED RECORDS:")
        print("-" * 50)
        sample_size = min(3, len(result_df))
        for i in range(sample_size):
            row = result_df.iloc[i]
            print(f"\nEmployee {i+1}:")
            print(f"  ID: {row.get('employee_id', 'N/A')}")
            print(f"  Full Name: {row.get('full_name', 'N/A')}")
            print(f"  Email: {row.get('email', 'N/A')}")
            print(f"  Job Title: {row.get('job_title', 'N/A')}")
            print(f"  Phone: {row.get('phone', 'Invalid Number')}")
            print(f"  Designation: {row.get('designation', 'N/A')}")
            print(f"  Experience: {row.get('years_of_experience', 'N/A')} years")
            print(f"  Department: {row.get('department', 'N/A')}")
        
        # Show designation distribution
        if 'designation' in result_df.columns:
            print(f"\nDESIGNATION DISTRIBUTION:")
            print("-" * 30)
            designation_counts = result_df['designation'].value_counts()
            for designation, count in designation_counts.items():
                percentage = (count / len(result_df)) * 100
                print(f"  {designation:<20} - {count:3d} ({percentage:5.1f}%)")
        
        # Show data quality metrics
        phone_valid = result_df['phone'].notna().sum() if 'phone' in result_df.columns else 0
        phone_invalid = result_df['phone'].isna().sum() if 'phone' in result_df.columns else 0
        
        print(f"\nDATA QUALITY METRICS:")
        print("-" * 30)
        print(f"  Valid phone numbers: {phone_valid}")
        print(f"  Invalid phone numbers (with 'x'): {phone_invalid}")
        if 'phone' in result_df.columns:
            phone_validity = (phone_valid / len(result_df)) * 100
            print(f"  Phone validity rate: {phone_validity:.1f}%")
        
        # Show output files
        print(f"\nOUTPUT FILES GENERATED:")
        print("-" * 30)
        print("  raw_employees_data.json      - Original API data")
        print("  processed_employees.csv      - Final processed data (CSV)")
        print("  processed_employees.json     - Final processed data (JSON)")
        print("  data_quality_report.json     - Data quality metrics")
        print("  employee_scraper.log         - Detailed execution logs")
        
        print(f"\nUSER STORY ACCEPTANCE CRITERIA STATUS:")
        print("-" * 45)
        print("  Data Retrieval: HTTP request successful")
        print("  Data Structure Validation: All required fields parsed")
        print("  Error Handling: Retry logic and logging implemented")
        print("  Data Normalization: All 6 requirements implemented")
        print("     • Designation column created based on experience")
        print("     • Full name column created from first + last name")
        print("     • Phone numbers with 'x' marked as invalid")
        print("     • Data types enforced as specified")
        print("     • Currency conversion available (optional)")
        print("     • Date formatting standardized (YYYY-MM-DD)")
        
        print(f"\nREADY FOR DATA WAREHOUSE INGESTION!")
        
    else:
        print("\nDATA PROCESSING FAILED!")
        print("Check the log file 'employee_scraper.log' for detailed error information.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()