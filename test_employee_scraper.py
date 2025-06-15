import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import requests
import json
from employee_scraper import EmployeeScraper

class TestEmployeeScraper(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_url = "https://api.slingacademy.com/v1/sample-data/files/employees.json"
        self.scraper = EmployeeScraper(self.api_url, max_retries=2, timeout=10)
        
        # Sample test data
        self.sample_data = [
            {
                "id": 1,
                "first_name": "John",
                "last_name": "Doe",
                "email": "john.doe@example.com",
                "job_title": "Software Engineer",
                "phone": "123-456-7890",
                "gender": "Male",
                "age": 30,
                "years_of_experience": 5,
                "salary": 75000,
                "department": "Engineering",
                "hire_date": "2020-01-15"
            },
            {
                "id": 2,
                "first_name": "Jane",
                "last_name": "Smith",
                "email": "jane.smith@example.com",
                "job_title": "Data Analyst",
                "phone": "987-654-x321",  # Invalid phone with 'x'
                "gender": "Female",
                "age": 25,
                "years_of_experience": 2,
                "salary": 60000,
                "department": "Analytics"
            }
        ]
    
    @patch('requests.get')
    def test_fetch_data_success(self, mock_get):
        """Test Case 1: Verify JSON File Download - Success"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_data
        mock_get.return_value = mock_response
        
        result = self.scraper.fetch_data()
        
        self.assertEqual(result, self.sample_data)
        mock_get.assert_called_once_with(
            self.api_url,
            timeout=10,
            headers={'User-Agent': 'Employee-Scraper/1.0'}
        )
    
    @patch('requests.get')
    def test_fetch_data_http_error(self, mock_get):
        """Test Case 1: Verify JSON File Download - HTTP Error"""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response
        
        result = self.scraper.fetch_data()
        
        self.assertIsNone(result)
    
    @patch('requests.get')
    def test_fetch_data_timeout_error(self, mock_get):
        """Test Case 1: Verify JSON File Download - Timeout Error"""
        # Mock timeout error
        mock_get.side_effect = requests.exceptions.Timeout()
        
        result = self.scraper.fetch_data()
        
        self.assertIsNone(result)
        self.assertEqual(mock_get.call_count, 2)  # Should retry
    
    @patch('requests.get')
    def test_fetch_data_connection_error(self, mock_get):
        """Test Case 1: Verify JSON File Download - Connection Error"""
        # Mock connection error
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        result = self.scraper.fetch_data()
        
        self.assertIsNone(result)
        self.assertEqual(mock_get.call_count, 2)  # Should retry
    
    def test_validate_data_structure_valid(self):
        """Test Case 4: Validate Data Structure - Valid Data"""
        result = self.scraper.validate_data_structure(self.sample_data)
        self.assertTrue(result)
    
    def test_validate_data_structure_invalid_not_list(self):
        """Test Case 4: Validate Data Structure - Invalid (Not List)"""
        invalid_data = {"error": "Invalid data"}
        result = self.scraper.validate_data_structure(invalid_data)
        self.assertFalse(result)
    
    def test_validate_data_structure_empty_list(self):
        """Test Case 4: Validate Data Structure - Empty List"""
        empty_data = []
        result = self.scraper.validate_data_structure(empty_data)
        self.assertFalse(result)
    
    def test_validate_data_structure_missing_fields(self):
        """Test Case 5: Handle Missing or Invalid Data"""
        incomplete_data = [{"id": 1, "first_name": "John"}]  # Missing required fields
        result = self.scraper.validate_data_structure(incomplete_data)
        self.assertFalse(result)
    
    def test_normalize_phone_number_valid(self):
        """Test phone number normalization - Valid"""
        valid_phone = "123-456-7890"
        result = self.scraper.normalize_phone_number(valid_phone)
        self.assertEqual(result, 1234567890)
    
    def test_normalize_phone_number_invalid_with_x(self):
        """Test phone number normalization - Invalid with 'x'"""
        invalid_phone = "123-456-x890"
        result = self.scraper.normalize_phone_number(invalid_phone)
        self.assertIsNone(result)
    
    def test_normalize_phone_number_empty(self):
        """Test phone number normalization - Empty"""
        result = self.scraper.normalize_phone_number("")
        self.assertIsNone(result)
    
    def test_determine_designation(self):
        """Test designation determination based on experience"""
        test_cases = [
            (1, "System Engineer"),
            (3, "Data Engineer"),
            (5, "Senior Data Engineer"),
            (7, "Senior Data Engineer"),
            (10, "Senior Data Engineer"),
            (15, "Lead")
        ]
        
        for years, expected in test_cases:
            with self.subTest(years=years):
                result = self.scraper.determine_designation(years)
                self.assertEqual(result, expected)
    
    def test_format_date_valid(self):
        """Test date formatting - Valid dates"""
        test_cases = [
            ("2020-01-15", "2020-01-15"),
            ("01/15/2020", "2020-01-15"),
            ("15/01/2020", "2020-01-15")
        ]
        
        for input_date, expected in test_cases:
            with self.subTest(input_date=input_date):
                result = self.scraper.format_date(input_date)
                self.assertEqual(result, expected)
    
    def test_format_date_invalid(self):
        """Test date formatting - Invalid date"""
        invalid_date = "invalid-date"
        result = self.scraper.format_date(invalid_date)
        self.assertIsNone(result)
    
    def test_normalize_data(self):
        """Test Case 2: Verify JSON File Extraction and Test Case 3: Validate File Type and Format"""
        df = self.scraper.normalize_data(self.sample_data)
        
        # Check if all expected columns exist
        expected_columns = [
            'id', 'full_name', 'email', 'phone', 'gender', 'age',
            'job_title', 'years_of_experience', 'salary', 'department', 'designation'
        ]
        
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check full name creation
        self.assertEqual(df.iloc[0]['full_name'], "John Doe")
        self.assertEqual(df.iloc[1]['full_name'], "Jane Smith")
        
        # Check designation assignment
        self.assertEqual(df.iloc[0]['designation'], "Senior Data Engineer")  # 5 years
        self.assertEqual(df.iloc[1]['designation'], "System Engineer")  # 2 years
        
        # Check phone number normalization
        self.assertEqual(df.iloc[0]['phone'], 1234567890)
        self.assertTrue(pd.isna(df.iloc[1]['phone']))  # Should be NaN due to 'x'
        
        # Check data types
        self.assertEqual(df['full_name'].dtype, 'string')
        self.assertEqual(df['email'].dtype, 'string')
        self.assertEqual(df['age'].dtype, 'int64')
    
    @patch.object(EmployeeScraper, 'fetch_data')
    @patch.object(EmployeeScraper, 'validate_data_structure')
    def test_process_employees_success(self, mock_validate, mock_fetch):
        """Test complete employee processing - Success"""
        mock_fetch.return_value = self.sample_data
        mock_validate.return_value = True
        
        result = self.scraper.process_employees()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        mock_fetch.assert_called_once()
        mock_validate.assert_called_once_with(self.sample_data)
    
    @patch.object(EmployeeScraper, 'fetch_data')
    def test_process_employees_fetch_failure(self, mock_fetch):
        """Test complete employee processing - Fetch Failure"""
        mock_fetch.return_value = None
        
        result = self.scraper.process_employees()
        
        self.assertIsNone(result)
        mock_fetch.assert_called_once()
    
    @patch.object(EmployeeScraper, 'fetch_data')
    @patch.object(EmployeeScraper, 'validate_data_structure')
    def test_process_employees_validation_failure(self, mock_validate, mock_fetch):
        """Test complete employee processing - Validation Failure"""
        mock_fetch.return_value = self.sample_data
        mock_validate.return_value = False
        
        result = self.scraper.process_employees()
        
        self.assertIsNone(result)

class TestEmployeeScraperIntegration(unittest.TestCase):
    """Integration tests that test the actual API (optional - can be skipped in CI)"""
    
    def setUp(self):
        self.api_url = "https://api.slingacademy.com/v1/sample-data/files/employees.json"
        self.scraper = EmployeeScraper(self.api_url)
    
    @unittest.skip("Skip integration test by default")
    def test_real_api_integration(self):
        """Integration test with real API - uncomment @unittest.skip to run"""
        result = self.scraper.process_employees()
        
        if result is not None:
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)
            
            # Check required columns exist
            required_columns = ['full_name', 'email', 'designation']
            for col in required_columns:
                self.assertIn(col, result.columns)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)