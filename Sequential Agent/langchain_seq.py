"""
CSV Data Analysis Multi-Agent Workflow
User provides CSV -> Analysis -> Code Execution -> Error Correction -> Results
Executes real Python code using Gemini Code Execution
"""

from google import genai
from google.genai import types
import os

# API Key - Paste directly here
GEMINI_API_KEY = "your_api_key_here"

# ============================================
# 1. DATA LOADING AGENT
# ============================================
class DataLoadingAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def load_csv(self, csv_file_path: str) -> dict:
        """Read CSV file and perform initial analysis"""
        print(f"📂 Reading CSV file: {csv_file_path}")
        
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()
            
            file_size = len(csv_content)
            lines = csv_content.split('\n')
            
            print(f"✓ File loaded")
            print(f"  Size: {file_size} bytes")
            print(f"  Number of lines: {len(lines)}")
            
            return {
                "file_path": csv_file_path,
                "content": csv_content,
                "size": file_size,
                "lines": len(lines),
                "preview": '\n'.join(lines[:5])
            }
        except Exception as e:
            print(f"❌ File loading error: {e}")
            return None

# ============================================
# 2. ANALYSIS AGENT - Analysis with Google Search
# ============================================
class AnalysisAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def analyze_data(self, csv_data: dict) -> dict:
        """Analyze the structure and content of CSV data"""
        print("🔍 Analyzing data...")
        
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])
        
        preview = csv_data['preview']
        
        prompt = f"""
By examining the first 5 rows of this CSV file:
1. Identify column names and types
2. Explain the data structure
3. Suggest what kind of analysis can be performed

CSV Data:
{preview}

Present the analysis in a brief and structured format.
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        
        analysis_text = response.text
        
        return {
            "analysis": analysis_text,
            "preview": preview,
            "file_size": csv_data['size']
        }

# ============================================
# 3. CODE GENERATION AGENT - With Code Execution
# ============================================
class CodeGenerationAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def generate_and_execute_code(self, csv_data: dict) -> dict:
        """Generate and execute Python code with CSV data"""
        print("💻 Generating and executing code...")
        
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )
        
        # Put CSV content directly into prompt
        csv_content = csv_data['content']
        
        prompt = f"""
Analyze this CSV data:

```csv
{csv_content}
```

TASK: Write Python code and perform the following:
1. Load the given CSV data as a DataFrame
2. Calculate basic statistics (mean, max, min, count)
3. Determine column types
4. Check for missing data
5. Show data distribution
6. Print all results in detail

CODE TO WRITE:
```python
import pandas as pd
import io

csv_data = \"\"\"
{csv_content}
\"\"\"

df = pd.read_csv(io.StringIO(csv_data.strip()))

print("=== DATA HEAD ===")
print(df.head(10))
print("\\n=== SHAPE ===")
print(f"Rows: {{len(df)}}, Columns: {{len(df.columns)}}")
print("\\n=== COLUMN TYPES ===")
print(df.dtypes)
print("\\n=== STATISTICS ===")
print(df.describe())
print("\\n=== MISSING DATA ===")
print(df.isnull().sum())
print("\\n=== COLUMN NAMES ===")
print(list(df.columns))
```

Execute the code and show all output.
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        
        # Extract results
        code_output = ""
        generated_code = ""
        execution_error = None
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                code_output += part.text
            if hasattr(part, 'executable_code') and part.executable_code:
                generated_code = part.executable_code.code
            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                exec_result = part.code_execution_result
                if exec_result.outcome == "OUTCOME_OK":
                    print(f"✓ Code executed successfully")
                else:
                    execution_error = exec_result.output
                    print(f"⚠️ Execution error occurred")
        
        return {
            "code": generated_code,
            "output": code_output,
            "execution_error": execution_error,
            "success": execution_error is None
        }

# ============================================
# 4. ERROR CORRECTION AGENT
# ============================================
class ErrorCorrectionAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def correct_and_retry(self, error_data: dict, csv_data: dict) -> dict:
        """Analyze error, fix code and retry execution"""
        print("🔧 Fixing error...")
        
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )
        
        csv_content = csv_data['content']
        
        prompt = f"""
Previous code caused an error:

ERROR:
{error_data['execution_error']}

Write CORRECTED code for this CSV data:

```csv
{csv_content}
```

TASK: Perform analysis with the following code:

```python
import pandas as pd
import io

csv_data = \"\"\"
{csv_content}
\"\"\"

try:
    df = pd.read_csv(io.StringIO(csv_data.strip()))
    
    print("=== SUCCESSFUL LOAD ===")
    print(f"Number of rows: {{len(df)}}")
    print(f"Number of columns: {{len(df.columns)}}")
    print("\\n=== DATA PREVIEW ===")
    print(df.head(10))
    print("\\n=== BASIC INFO ===")
    print(df.info())
    print("\\n=== STATISTICS ===")
    print(df.describe())
    print("\\n=== MISSING DATA ===")
    print(df.isnull().sum())
    
except Exception as e:
    print(f"ERROR: {{e}}")
```

Execute the code and show results completely.
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        
        corrected_code = ""
        output = ""
        error = None
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                output += part.text
            if hasattr(part, 'executable_code') and part.executable_code:
                corrected_code = part.executable_code.code
            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                if part.code_execution_result.outcome != "OUTCOME_OK":
                    error = part.code_execution_result.output
        
        if error is None:
            print(f"✓ Error fixed and code executed successfully")
        
        return {
            "corrected_code": corrected_code,
            "output": output,
            "error": error,
            "success": error is None
        }

# ============================================
# 5. VISUALIZATION AGENT - Chart Creation
# ============================================
class VisualizationAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def create_visualizations(self, csv_data: dict) -> dict:
        """Create data visualizations"""
        print("📈 Creating visualizations...")
        
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )
        
        csv_content = csv_data['content']
        
        prompt = f"""
Visualize this CSV data:

```csv
{csv_content}
```

TASK: Create charts with Matplotlib and Seaborn:
1. Distribution of numerical columns (histogram)
2. Relationship graph (scatter plot) - if 2+ numerical columns exist
3. Bar chart for categorical data
4. Time series - if date column exists

CODE:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

csv_data = \"\"\"
{csv_content}
\"\"\"

df = pd.read_csv(io.StringIO(csv_data.strip()))

# Set style
sns.set_style("darkgrid")
plt.figure(figsize=(15, 10))

# Find numerical columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Create subplot based on number of columns
n_cols = len(numeric_cols)
if n_cols > 0:
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        plt.hist(df[col].dropna(), bins=20, edgecolor='black')
        plt.title(f'{{col}} Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('data_visualization.png', dpi=100, bbox_inches='tight')
print("✓ Chart saved: data_visualization.png")
plt.show()
```

Execute the code.
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        
        output = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                output += part.text
        
        print("✓ Visualization completed")
        return {"visualization_output": output}

# ============================================
# 6. INSIGHT AGENT - Deep Insights
# ============================================
class InsightAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def extract_insights(self, analysis_output: str, csv_preview: str) -> dict:
        """Find hidden patterns and insights in the data"""
        print("🔎 Extracting insights...")
        
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])
        
        prompt = f"""
Extract VITAL insights and patterns from this data analysis results:

{analysis_output}

Data Preview:
{csv_preview}

TASK:
1. Identify the top 3 most important findings
2. Describe statistical anomalies
3. Provide business recommendations (data-driven)
4. Identify potential risks and opportunities
5. Research relevant industry trends

Provide the answer briefly and in a structured format.
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        
        insights = response.text
        print("✓ Insights extracted")
        return {"insights": insights}

# ============================================
# 7. ANOMALY DETECTION AGENT - Anomaly Detection
# ============================================
class AnomalyDetectionAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def detect_anomalies(self, csv_data: dict) -> dict:
        """Analyze data for abnormal values"""
        print("🚨 Detecting anomalies...")
        
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )
        
        csv_content = csv_data['content']
        
        prompt = f"""
Detect abnormal and suspicious values in this CSV data:

```csv
{csv_content}
```

TASK: Write Python code and:
1. Detect outliers using Z-score
2. Find anomalies using IQR method
3. Identify missing data
4. Check for duplicate rows
5. Find data type inconsistencies

CODE:
```python
import pandas as pd
import numpy as np
import io
from scipy import stats

csv_data = \"\"\"
{csv_content}
\"\"\"

df = pd.read_csv(io.StringIO(csv_data.strip()))

print("=== ANOMALY DETECTION ===")
print(f"Total Rows: {{len(df)}}")
print(f"Duplicate Rows: {{df.duplicated().sum()}}")

# Anomalies for numerical columns
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outliers = (z_scores > 3).sum()
    print(f"\\n{{col}} - Outliers (Z-score > 3): {{outliers}}")
    
print(f"\\nTotal Missing Data: {{df.isnull().sum().sum()}}")
```

Execute the code.
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        
        anomaly_output = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                anomaly_output += part.text
        
        print("✓ Anomaly detection completed")
        return {"anomalies": anomaly_output}

# ============================================
# 8. RECOMMENDATION AGENT - Recommendations and Actions
# ============================================
class RecommendationAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def generate_recommendations(self, insights: str, anomalies: str) -> dict:
        """Recommend actions that should be taken"""
        print("💡 Generating recommendations...")
        
        prompt = f"""
Create an ACTION PLAN based on this analysis data:

INSIGHTS:
{insights}

ANOMALIES:
{anomalies}

TASK:
1. 3 urgent actions to be taken
2. Impact/difficulty score for each action
3. Implementation timeline
4. Define success metrics
5. Risk mitigation plan

Provide the answer in bullet points and in an actionable format.
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[{"text": prompt}]
        )
        
        recommendations = response.text
        print("✓ Recommendations generated")
        return {"recommendations": recommendations}

# ============================================
# 9. FINAL REPORT AGENT - Comprehensive Report
# ============================================
class FinalReportAgent:
    def __init__(self, api_key: str):
        self.llm = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def generate_report(self, all_results: dict) -> str:
        """Generate executive summary report"""
        print("📊 Generating final report...")
        
        return f"""
╔═══════════════════════════════════════════════════════════════╗
║        COMPREHENSIVE DATA ANALYSIS REPORT - EXECUTIVE SUMMARY ║
╚═══════════════════════════════════════════════════════════════╝

1️⃣ BASIC STATISTICS
{'─'*60}
{all_results.get('code_result', {}).get('output', 'N/A')[:500]}...

2️⃣ ANOMALY DETECTION
{'─'*60}
{all_results.get('anomaly_result', {}).get('anomalies', 'N/A')[:400]}...

3️⃣ DEEP INSIGHTS
{'─'*60}
{all_results.get('insight_result', {}).get('insights', 'N/A')[:400]}...

4️⃣ VISUALIZATION
{'─'*60}
✓ Charts created: data_visualization.png

5️⃣ RECOMMENDED ACTIONS
{'─'*60}
{all_results.get('recommendation_result', {}).get('recommendations', 'N/A')[:400]}...

{'='*60}
✅ REPORT COMPLETED - {len(all_results)} DIFFERENT ANALYSES PERFORMED
{'='*60}
"""

# ============================================
# 10. MAIN WORKFLOW
# ============================================
def run_csv_analysis_workflow(csv_file_path: str):
    """Run CSV data analysis workflow"""
    
    print(f"\n{'='*70}")
    print(f"🚀 COMPREHENSIVE CSV DATA ANALYSIS WORKFLOW STARTING")
    print(f"{'='*70}\n")
    
    all_results = {}
    
    # STEP 1: Data Loading
    print("📂 STEP 1: DATA LOADING")
    print(f"{'─'*70}")
    loader = DataLoadingAgent(GEMINI_API_KEY)
    csv_data = loader.load_csv(csv_file_path)
    
    if not csv_data:
        print("❌ File could not be loaded!")
        return
    
    # STEP 2: Analysis
    print("\n🔍 STEP 2: DATA ANALYSIS")
    print(f"{'─'*70}")
    analyzer = AnalysisAgent(GEMINI_API_KEY)
    analysis_result = analyzer.analyze_data(csv_data)
    print(f"✓ Analysis completed\n")
    all_results['analysis_result'] = analysis_result
    
    # STEP 3: Code Execution
    print("💻 STEP 3: CODE EXECUTION & STATISTICS")
    print(f"{'─'*70}")
    code_gen = CodeGenerationAgent(GEMINI_API_KEY)
    code_result = code_gen.generate_and_execute_code(csv_data)
    all_results['code_result'] = code_result
    
    # STEP 4: Error Correction (if needed)
    if not code_result['success']:
        print("\n🔧 STEP 4: ERROR CORRECTION")
        print(f"{'─'*70}")
        corrector = ErrorCorrectionAgent(GEMINI_API_KEY)
        corrected_result = corrector.correct_and_retry(code_result, csv_data)
        all_results['code_result'] = corrected_result
    
    # STEP 5: Visualization
    print("\n📈 STEP 5: VISUALIZATION")
    print(f"{'─'*70}")
    viz_agent = VisualizationAgent(GEMINI_API_KEY)
    viz_result = viz_agent.create_visualizations(csv_data)
    all_results['viz_result'] = viz_result
    
    # STEP 6: Anomaly Detection
    print("\n🚨 STEP 6: ANOMALY DETECTION")
    print(f"{'─'*70}")
    anomaly_agent = AnomalyDetectionAgent(GEMINI_API_KEY)
    anomaly_result = anomaly_agent.detect_anomalies(csv_data)
    all_results['anomaly_result'] = anomaly_result
    
    # STEP 7: Insight Extraction
    print("\n🔎 STEP 7: DEEP INSIGHTS")
    print(f"{'─'*70}")
    insight_agent = InsightAgent(GEMINI_API_KEY)
    insight_result = insight_agent.extract_insights(
        code_result['output'], 
        analysis_result['preview']
    )
    all_results['insight_result'] = insight_result
    
    # STEP 8: Recommendations
    print("\n💡 STEP 8: RECOMMENDATIONS AND ACTION PLAN")
    print(f"{'─'*70}")
    rec_agent = RecommendationAgent(GEMINI_API_KEY)
    rec_result = rec_agent.generate_recommendations(
        insight_result['insights'],
        anomaly_result['anomalies']
    )
    all_results['recommendation_result'] = rec_result
    
    # STEP 9: Final Report
    print("\n📊 STEP 9: FINAL REPORT")
    print(f"{'─'*70}")
    reporter = FinalReportAgent(GEMINI_API_KEY)
    final_report = reporter.generate_report(all_results)
    
    print(final_report)

# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║   CSV DATA ANALYSIS MULTI-AGENT WORKFLOW                   ║")
    print("║   Real Analysis with Gemini Code Execution                ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Enter CSV file path
    csv_file = input("\n📂 Enter CSV file path (e.g., data.csv): ").strip()
    
    if not csv_file:
        # Create test data
        print("\n📝 Creating test CSV file...")
        test_csv = """Name,Age,Salary,Department
Ali,28,5000,IT
Ayse,32,6000,HR
Mehmet,25,4500,IT
Fatma,29,5500,Sales
Can,31,6500,IT
Zeynep,27,5200,Marketing
"""
        with open("test_data.csv", "w", encoding='utf-8') as f:
            f.write(test_csv)
        csv_file = "test_data.csv"
        print(f"✓ Test file created: {csv_file}")
    
    run_csv_analysis_workflow(csv_file)