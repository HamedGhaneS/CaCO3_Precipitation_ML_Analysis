#!/usr/bin/env python3
"""
ML-based mapping from Xi to Yi for Ca and Na data analysis
File path: H:\FreeLancer\Elmira.xlsx

Author: ML Analysis Script
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class MLAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.ca_data = None
        self.na_data = None
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_and_process_data(self):
        """Load and process Excel data for ML analysis"""
        print("Loading and processing data...")
        
        # Load both sheets
        sheet1 = pd.read_excel(self.file_path, sheet_name='One stage')
        sheet2 = pd.read_excel(self.file_path, sheet_name='Two-stage')
        
        print(f"Loaded Sheet 1: {sheet1.shape}, Sheet 2: {sheet2.shape}")
        
        # Process Sheet 2 (has alkali type information)
        processed_data = []
        
        for idx, row in sheet2.iterrows():
            if idx == 0:  # Skip header
                continue
                
            # Extract input features (X)
            try:
                x1 = row.iloc[2]  # Ca concentration
                x2 = row.iloc[3]  # time
                x3 = row.iloc[4]  # pH
                x4 = row.iloc[5]  # CO2 flow
                alkali_type = row.iloc[1]  # Alkali type
                
                # Skip if any X variable is missing
                if pd.isna([x1, x2, x3, x4]).any():
                    continue
                
                # Extract Y1 values (Peak temperature) - columns 6-12
                y1_values = [val for val in row.iloc[6:13] if pd.notna(val)]
                
                # Extract Y2 values (CO2 content) - columns 13-19  
                y2_values = [val for val in row.iloc[13:20] if pd.notna(val)]
                
                # Extract Y3 values (Calcite content) - columns 20+
                y3_values = [val for val in row.iloc[20:] if pd.notna(val)]
                
                # Only include if we have at least one Y measurement
                if len(y1_values) > 0 or len(y2_values) > 0 or len(y3_values) > 0:
                    data_point = {
                        'x1_ca_conc': x1,
                        'x2_time': x2,
                        'x3_ph': x3,
                        'x4_co2_flow': x4,
                        'y1_peak_temp_mean': np.mean(y1_values) if y1_values else np.nan,
                        'y1_peak_temp_std': np.std(y1_values) if len(y1_values) > 1 else 0,
                        'y2_co2_content_mean': np.mean(y2_values) if y2_values else np.nan,
                        'y2_co2_content_std': np.std(y2_values) if len(y2_values) > 1 else 0,
                        'y3_calcite_content_mean': np.mean(y3_values) if y3_values else np.nan,
                        'y3_calcite_content_std': np.std(y3_values) if len(y3_values) > 1 else 0,
                        'alkali_type': alkali_type,
                        'n_y1_measurements': len(y1_values),
                        'n_y2_measurements': len(y2_values),
                        'n_y3_measurements': len(y3_values)
                    }
                    processed_data.append(data_point)
                    
            except Exception as e:
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        # Separate Ca and Na data
        self.ca_data = df[df['alkali_type'] == 'Ca'].copy()
        self.na_data = df[df['alkali_type'] == 'Na'].copy()
        
        print(f"Processed data - Ca samples: {len(self.ca_data)}, Na samples: {len(self.na_data)}")
        
        return df
    
    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        for name, data in [('Calcium (Ca)', self.ca_data), ('Sodium (Na)', self.na_data)]:
            if len(data) == 0:
                continue
                
            print(f"\n{name} Data Statistics:")
            print(f"Samples: {len(data)}")
            
            # Input features statistics
            print("Input Features (X):")
            print(f"  X1 (Ca conc): {data['x1_ca_conc'].min():.0f} - {data['x1_ca_conc'].max():.0f} ppm")
            print(f"  X2 (time): {data['x2_time'].min():.0f} - {data['x2_time'].max():.0f} min")
            print(f"  X3 (pH): {data['x3_ph'].min():.1f} - {data['x3_ph'].max():.1f}")
            print(f"  X4 (CO2 flow): {data['x4_co2_flow'].min():.1f} - {data['x4_co2_flow'].max():.1f} L/M")
            
            # Output targets statistics
            print("Output Targets (Y):")
            for col, desc in [('y1_peak_temp_mean', 'Peak Temperature (°C)'), 
                            ('y2_co2_content_mean', 'CO2 Content (%)'),
                            ('y3_calcite_content_mean', 'Calcite Content (%)')]:
                valid_data = data[col].dropna()
                if len(valid_data) > 0:
                    print(f"  {desc}: {len(valid_data)} samples, {valid_data.min():.3f} - {valid_data.max():.3f}")
                else:
                    print(f"  {desc}: No valid data")
    
    def create_visualization(self):
        """Create visualizations of the data"""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Xi to Yi Mapping Analysis: Ca vs Na', fontsize=16)
        
        # Plot Y1, Y2, Y3 for both Ca and Na
        targets = ['y1_peak_temp_mean', 'y2_co2_content_mean', 'y3_calcite_content_mean']
        target_names = ['Peak Temperature (°C)', 'CO2 Content (%)', 'Calcite Content (%)']
        
        for i, (target, name) in enumerate(zip(targets, target_names)):
            # Ca data
            ca_valid = self.ca_data[target].dropna()
            na_valid = self.na_data[target].dropna()
            
            axes[0, i].scatter(range(len(ca_valid)), ca_valid, alpha=0.7, color='blue', label='Ca')
            axes[0, i].set_title(f'Ca - {name}')
            axes[0, i].set_xlabel('Sample Index')
            axes[0, i].set_ylabel(name)
            axes[0, i].grid(True, alpha=0.3)
            
            # Na data
            axes[1, i].scatter(range(len(na_valid)), na_valid, alpha=0.7, color='red', label='Na')
            axes[1, i].set_title(f'Na - {name}')
            axes[1, i].set_xlabel('Sample Index')
            axes[1, i].set_ylabel(name)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('H:/FreeLancer/Elmira/data_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_ml_models(self, data, data_name):
        """Train multiple ML models for Xi -> Yi mapping"""
        print(f"\n=== Training ML Models for {data_name} ===")
        
        if len(data) < 5:
            print(f"Insufficient data for {data_name} (need at least 5 samples)")
            return
        
        # Define input features
        X_features = ['x1_ca_conc', 'x2_time', 'x3_ph', 'x4_co2_flow']
        X = data[X_features].values
        
        # Define target variables
        targets = {
            'Y1_Peak_Temp': 'y1_peak_temp_mean',
            'Y2_CO2_Content': 'y2_co2_content_mean', 
            'Y3_Calcite_Content': 'y3_calcite_content_mean'
        }
        
        # ML algorithms to test
        algorithms = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0),
            'Lasso_Regression': Lasso(alpha=0.1),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        results = {}
        
        for target_name, target_col in targets.items():
            print(f"\nTraining models for {target_name}:")
            
            # Get valid data for this target
            valid_data = data[data[target_col].notna()]
            if len(valid_data) < 5:
                print(f"  Insufficient data for {target_name} ({len(valid_data)} samples)")
                continue
                
            X_target = valid_data[X_features].values
            y_target = valid_data[target_col].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_target)
            
            target_results = {}
            
            for alg_name, algorithm in algorithms.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(algorithm, X_scaled, y_target, 
                                              cv=min(5, len(valid_data)), 
                                              scoring='r2')
                    
                    # Train on full data
                    algorithm.fit(X_scaled, y_target)
                    y_pred = algorithm.predict(X_scaled)
                    
                    # Calculate metrics
                    r2 = r2_score(y_target, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_target, y_pred))
                    mae = mean_absolute_error(y_target, y_pred)
                    
                    target_results[alg_name] = {
                        'R2_Score': r2,
                        'CV_R2_Mean': cv_scores.mean(),
                        'CV_R2_Std': cv_scores.std(),
                        'RMSE': rmse,
                        'MAE': mae,
                        'Model': algorithm,
                        'Scaler': scaler
                    }
                    
                    print(f"  {alg_name}: R² = {r2:.3f}, CV R² = {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                    
                except Exception as e:
                    print(f"  {alg_name}: Failed - {str(e)}")
                    continue
            
            results[target_name] = target_results
            
        self.results[data_name] = results
        return results
    
    def create_prediction_plots(self, data, data_name, results):
        """Create prediction vs actual plots"""
        if not results:
            return
            
        targets = list(results.keys())
        n_targets = len(targets)
        
        if n_targets == 0:
            return
        
        fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))
        if n_targets == 1:
            axes = [axes]
            
        fig.suptitle(f'{data_name} - Predictions vs Actual', fontsize=14)
        
        target_cols = {
            'Y1_Peak_Temp': 'y1_peak_temp_mean',
            'Y2_CO2_Content': 'y2_co2_content_mean', 
            'Y3_Calcite_Content': 'y3_calcite_content_mean'
        }
        
        X_features = ['x1_ca_conc', 'x2_time', 'x3_ph', 'x4_co2_flow']
        
        for i, target_name in enumerate(targets):
            ax = axes[i]
            target_col = target_cols[target_name]
            
            # Get best model (highest R2)
            best_model_name = max(results[target_name].keys(), 
                                key=lambda x: results[target_name][x]['R2_Score'])
            best_result = results[target_name][best_model_name]
            
            # Get valid data
            valid_data = data[data[target_col].notna()]
            X_target = valid_data[X_features].values
            y_actual = valid_data[target_col].values
            
            # Make predictions
            X_scaled = best_result['Scaler'].transform(X_target)
            y_pred = best_result['Model'].predict(X_scaled)
            
            # Plot
            ax.scatter(y_actual, y_pred, alpha=0.7)
            ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{target_name}\nBest Model: {best_model_name}\nR² = {best_result["R2_Score"]:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        filename = f'H:/FreeLancer/Elmira/{data_name.lower()}_predictions.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*60)
        print("ML ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        for data_name, results in self.results.items():
            print(f"\n{data_name.upper()} RESULTS:")
            print("-" * 40)
            
            if not results:
                print("No valid models trained")
                continue
                
            for target, algorithms in results.items():
                print(f"\n{target}:")
                if not algorithms:
                    print("  No valid models")
                    continue
                    
                # Find best model
                best_model = max(algorithms.keys(), 
                               key=lambda x: algorithms[x]['R2_Score'])
                best_result = algorithms[best_model]
                
                print(f"  Best Model: {best_model}")
                print(f"  R² Score: {best_result['R2_Score']:.4f}")
                print(f"  Cross-Val R²: {best_result['CV_R2_Mean']:.4f} ± {best_result['CV_R2_Std']:.4f}")
                print(f"  RMSE: {best_result['RMSE']:.4f}")
                print(f"  MAE: {best_result['MAE']:.4f}")
                
                print(f"  All Models Performance:")
                for alg_name, result in algorithms.items():
                    print(f"    {alg_name}: R² = {result['R2_Score']:.3f}")
    
    def save_results_to_excel(self):
        """Save results to Excel file"""
        print("\nSaving results to Excel...")
        
        with pd.ExcelWriter('H:/FreeLancer/Elmira/ML_Analysis_Results.xlsx', engine='openpyxl') as writer:
            # Save processed data
            if self.ca_data is not None and len(self.ca_data) > 0:
                self.ca_data.to_excel(writer, sheet_name='Ca_Data', index=False)
            
            if self.na_data is not None and len(self.na_data) > 0:
                self.na_data.to_excel(writer, sheet_name='Na_Data', index=False)
            
            # Save model results summary
            summary_data = []
            for data_name, results in self.results.items():
                for target, algorithms in results.items():
                    for alg_name, result in algorithms.items():
                        summary_data.append({
                            'Dataset': data_name,
                            'Target': target,
                            'Algorithm': alg_name,
                            'R2_Score': result['R2_Score'],
                            'CV_R2_Mean': result['CV_R2_Mean'],
                            'CV_R2_Std': result['CV_R2_Std'],
                            'RMSE': result['RMSE'],
                            'MAE': result['MAE']
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Model_Results', index=False)
        
        print("Results saved to H:/FreeLancer/Elmira/ML_Analysis_Results.xlsx")
    
    def run_complete_analysis(self):
        """Run the complete ML analysis pipeline"""
        print("Starting ML Analysis for Xi -> Yi Mapping")
        print("="*50)
        
        # Load and process data
        self.load_and_process_data()
        
        # Exploratory data analysis
        self.exploratory_data_analysis()
        
        # Create visualizations
        self.create_visualization()
        
        # Train ML models for Ca data
        if len(self.ca_data) > 0:
            ca_results = self.train_ml_models(self.ca_data, "Calcium (Ca)")
            self.create_prediction_plots(self.ca_data, "Calcium_Ca", ca_results)
        
        # Train ML models for Na data  
        if len(self.na_data) > 0:
            na_results = self.train_ml_models(self.na_data, "Sodium (Na)")
            self.create_prediction_plots(self.na_data, "Sodium_Na", na_results)
        
        # Generate summary report
        self.generate_summary_report()
        
        # Save results
        self.save_results_to_excel()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("Check the generated files:")
        print("- ML_Analysis_Results.xlsx")
        print("- data_visualization.png")
        print("- calcium_ca_predictions.png")
        print("- sodium_na_predictions.png")


def main():
    """Main function to run the analysis"""
    # File path
    file_path = r"H:\FreeLancer\Elmira.xlsx"
    
    # Create analyzer instance
    analyzer = MLAnalyzer(file_path)
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()