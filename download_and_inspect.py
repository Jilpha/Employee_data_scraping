
import requests
import json
import os
from datetime import datetime

def download_and_inspect_json():
    """Download JSON from API and show detailed information"""
    
    api_url = "https://api.slingacademy.com/v1/sample-data/files/employees.json"
    
    print("="*60)
    print("DOWNLOADING AND INSPECTING JSON DATA")
    print("="*60)
    
    try:
        print(f" Fetching data from: {api_url}")
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            print(" Successfully downloaded data!")
            
            # Parse JSON
            data = response.json()
            
            # Save raw JSON with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"raw_employees_{timestamp}.json"
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            current_dir = os.getcwd()
            full_path = os.path.join(current_dir, json_filename)
            
            print(f" Raw JSON saved to: {full_path}")
            print(f" File size: {os.path.getsize(json_filename):,} bytes")
            
            # Inspect the data
            print("\n" + "="*40)
            print("DATA INSPECTION")
            print("="*40)
            
            if isinstance(data, list):
                print(f" Total records: {len(data)}")
                
                if len(data) > 0:
                    first_record = data[0]
                    print(f" Fields in each record: {len(first_record)}")
                    print(" Field names:")
                    for field in first_record.keys():
                        print(f"   - {field}")
                    
                    print(f"\n Sample record (first employee):")
                    for key, value in first_record.items():
                        print(f"   {key}: {value}")
                    
                    # Show a few more examples
                    print(f"\n Sample of employees:")
                    for i in range(min(5, len(data))):
                        emp = data[i]
                        name = f"{emp.get('first_name', 'N/A')} {emp.get('last_name', 'N/A')}"
                        experience = emp.get('years_of_experience', 'N/A')
                        department = emp.get('department', 'N/A')
                        print(f"   {i+1}. {name} - {experience} years - {department}")
                    
                    # Data quality check
                    print(f"\n DATA QUALITY CHECK:")
                    phone_with_x = sum(1 for emp in data if 'x' in str(emp.get('phone', '')).lower())
                    print(f"   Records with 'x' in phone: {phone_with_x}")
                    
                    departments = set(emp.get('department', 'Unknown') for emp in data)
                    print(f"   Unique departments: {len(departments)}")
                    print(f"   Departments: {', '.join(sorted(departments))}")
                    
                    experience_range = [emp.get('years_of_experience', 0) for emp in data if emp.get('years_of_experience') is not None]
                    if experience_range:
                        print(f"   Experience range: {min(experience_range)} - {max(experience_range)} years")
                        avg_exp = sum(experience_range) / len(experience_range)
                        print(f"   Average experience: {avg_exp:.1f} years")
            
            else:
                print(" Data is not in expected list format")
                print(f"Data type: {type(data)}")
                print(f"Data content: {data}")
            
            print(f"\n TO VIEW THE FILE:")
            print(f"   1. Open VS Code and look for: {json_filename}")
            print(f"   2. Or navigate to: {full_path}")
            print(f"   3. Or use command: code {json_filename}")
            
        else:
            print(f" Failed to download data. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f" Network error: {str(e)}")
    except json.JSONDecodeError as e:
        print(f" JSON parsing error: {str(e)}")
    except Exception as e:
        print(f" Unexpected error: {str(e)}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    download_and_inspect_json()
