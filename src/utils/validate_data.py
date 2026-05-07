import great_expectations as gx
import pandas as pd
from typing import Tuple, List

def validate_claims_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for DE-SynPUF Medicare Claims dataset using GX 1.0+.
    """
    print("🔍 Starting data validation with Great Expectations 1.0+...")
    
    # === 1. SETUP DATA CONTEXT & PIPELINE ===
    # Get the Data Context
    context = gx.get_context()
    
    # Create a pandas Data Source
    data_source = context.data_sources.add_pandas(name="claims_data_source")
    
    # Create a Data Asset
    data_asset = data_source.add_dataframe_asset(name="claims_data_asset")
    
    # Create a Batch Definition for the whole dataframe
    batch_definition = data_asset.add_batch_definition_whole_dataframe("claims_batch_definition")
    
    # Define the Batch Parameter dictionary and get the batch
    batch_parameters = {"dataframe": df}
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    # === 2. DEFINE EXPECTATIONS ===
    print("   📋 Compiling schema and business logic expectations...")
    expectations_to_run = []
    
    # Core identifiers must exist and not be null
    essential_cols = [
        'CLM_ID', 'DESYNPUF_ID', 'CLM_FROM_DT', 
        'CLM_THRU_DT', 'PRVDR_NUM', 'CLM_UTLZTN_DAY_CNT'
    ]
    for col in essential_cols:
        expectations_to_run.append(gx.expectations.ExpectColumnToExist(column=col))
        expectations_to_run.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))
    
    # Physician NPIs must exist, not be null, and be object/string type
    npi_cols = ['AT_PHYSN_NPI', 'OP_PHYSN_NPI', 'OT_PHYSN_NPI']
    for col in npi_cols:
        expectations_to_run.append(gx.expectations.ExpectColumnToExist(column=col))
        expectations_to_run.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))
        expectations_to_run.append(gx.expectations.ExpectColumnValuesToBeOfType(column=col, type_="object"))
        
    # Financial features schema & numeric range validation
    expectations_to_run.append(gx.expectations.ExpectColumnToExist(column="CLM_PMT_AMT"))
    expectations_to_run.append(gx.expectations.ExpectColumnToExist(column="NCH_PRMRY_PYR_CLM_PD_AMT"))
    expectations_to_run.append(gx.expectations.ExpectColumnValuesToBeBetween(column="CLM_UTLZTN_DAY_CNT", min_value=0))
    expectations_to_run.append(gx.expectations.ExpectColumnValuesToBeBetween(column="CLM_PMT_AMT", min_value=0))
    expectations_to_run.append(gx.expectations.ExpectColumnValuesToBeBetween(column="NCH_PRMRY_PYR_CLM_PD_AMT", min_value=0))
    
    # Data consistency: Claim Thru Date >= Claim From Date
    expectations_to_run.append(gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="CLM_THRU_DT",
        column_B="CLM_FROM_DT",
        or_equal=True,
        mostly=0.99
    ))

    # === 3. RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    
    failed_expectations = []
    passed_checks = 0
    total_checks = len(expectations_to_run)
    
    # Test individual Expectations against the batch
    for expectation in expectations_to_run:
        result = batch.validate(expectation)
        
        if result.success:
            passed_checks += 1
        else:
            # Capture the name of the Expectation that failed
            failed_expectations.append(type(expectation).__name__)
            
    is_success = passed_checks == total_checks

    # === 4. PROCESS RESULTS ===
    if is_success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        failed_checks = total_checks - passed_checks
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        # Using set() to show unique types of failed tests
        print(f"   Failed expectations: {list(set(failed_expectations))}")
    
    return is_success, failed_expectations

# Example usage:
# is_valid, failures = validate_claims_data(df_cleaned)