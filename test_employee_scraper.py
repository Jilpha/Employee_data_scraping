import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from employee_scraper import EmployeeScraper
import json

class TestEmployeeScraper(unittest.TestCase):
    def setUp(self):
        self.api_url = "https://api.slingacademy.com/v1/sample-data/files/employees.json"
        self.scraper = EmployeeScraper(self.api_url)
        self.sample_data = [{
            "id": 1,
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "job_title": "Engineer",
            "phone": "1234567890",
            "gender": "Male",
            "age": 30,
            "years_of_experience": 5,
            "salary": 70000,
            "department": "Engineering",
            "hire_date": "2022-01-15"
        }]

    @patch("employee_scraper.requests.get")
    def test_fetch_json_data(self, mock_get):
        """Test Case 1: Verify JSON File Download"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.sample_data
        data = self.scraper.fetch_data()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['first_name'], 'John')

    @patch("builtins.open", new_callable=mock_open)
    @patch("employee_scraper.json.dump")
    def test_save_raw_json_file(self, mock_json_dump, mock_file):
        """Test Case 2: Verify JSON File Extraction"""
        self.scraper.raw_data = self.sample_data
        df = pd.DataFrame(self.sample_data)
        report = {"summary": {}}

        self.scraper.save_results(df, report)

        # Check if raw_employees_data.json file was opened
        calls = mock_file.call_args_list
        expected_call = ('raw_employees_data.json', 'w')
        found = any(call.args[:2] == expected_call for call in calls)
        self.assertTrue(found, "Expected call to open raw_employees_data.json not found.")

    def test_validate_data_structure_success(self):
        """Test Case 3: Validate File Type and Format"""
        result = self.scraper.validate_data_structure(self.sample_data)
        self.assertTrue(result)

    def test_validate_data_structure_missing_field(self):
        """Test Case 4: Validate Data Structure"""
        invalid_data = [dict(self.sample_data[0])]
        del invalid_data[0]['email']
        result = self.scraper.validate_data_structure(invalid_data)
        self.assertFalse(result)

    def test_handle_invalid_phone_number(self):
        """Test Case 5: Handle phone with 'x' marked invalid"""
        sample = self.sample_data[0].copy()
        sample['phone'] = '123x456'
        parsed = self.scraper.parse_employee_fields([sample])
        df = self.scraper.normalize_data(parsed)
        self.assertTrue(pd.isna(df['phone'].iloc[0]))  # FIXED: pd.isna used instead of assertIsNone

    def test_parse_and_normalize(self):
        """Test: Ensure normalization returns expected columns"""
        parsed = self.scraper.parse_employee_fields(self.sample_data)
        df = self.scraper.normalize_data(parsed)
        expected_columns = [
            'employee_id', 'full_name', 'email', 'phone',
            'gender', 'age', 'job_title', 'years_of_experience',
            'salary', 'department', 'designation', 'hire_date'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_designation_assignment(self):
        """Test: Designation logic based on experience"""
        self.assertEqual(self.scraper.determine_designation(1), "System Engineer")
        self.assertEqual(self.scraper.determine_designation(4), "Data Engineer")
        self.assertEqual(self.scraper.determine_designation(7), "Senior Data Engineer")
        self.assertEqual(self.scraper.determine_designation(15), "Lead")
        self.assertEqual(self.scraper.determine_designation(None), "Unknown")

if __name__ == '__main__':
    unittest.main()
