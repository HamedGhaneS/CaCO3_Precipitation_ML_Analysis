#!/usr/bin/env python3
"""
ML Prediction Models for Y1, Y2, Y3 based on X1, X2, X3, X4 parameters
Creates comprehensive prediction models for Ca and Na systems with extended parameter bounds

File path: H:\FreeLancer\Elmira\DataSpreadsheet.xlsx
Repository: CaCO3_Precipitation_ML_Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

class YiPredictionAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.ca_data = None
        self.na_data = None
        self.parameter_bounds = {}
        self.best_models = {}
        self.prediction_grids = {}
        self.feature_importance = {}  # Store feature importance results
        
    def load_and_process_data(self):
        """Load and process data for prediction modeling"""
        print("Loading data for Yi prediction modeling...")
        
        sheet2 = pd.read_excel(self.file_path, sheet_name='Two-stage')
        processed_data = []
        
        for idx, row in sheet2.iterrows():
            if idx == 0:  # Skip header
                continue
                
            try:
                x1 = row.iloc[2]  # Ca concentration
                x2 = row.iloc[3]  # time
                x3 = row.iloc[4]  # pH
                x4 = row.iloc[5]  # CO2 flow
                alkali_type = row.iloc[1]  # Alkali type
                
                if pd.isna([x1, x2, x3, x4]).any():
                    continue
                
                # Extract Y values (multiple measurements per sample)
                y1_values = [val for val in row.iloc[6:13] if pd.notna(val)]
                y2_values = [val for val in row.iloc[13:20] if pd.notna(val)]
                y3_values = [val for val in row.iloc[20:] if pd.notna(val)]
                
                if len(y1_values) > 0 or len(y2_values) > 0 or len(y3_values) > 0:
                    data_point = {
                        'X1_Ca_Conc': x1,
                        'X2_Time': x2,
                        'X3_pH': x3,
                        'X4_CO2_Flow': x4,
                        'Y1_Peak_Temp': np.mean(y1_values) if y1_values else np.nan,
                        'Y2_CO2_Content': np.mean(y2_values) if y2_values else np.nan,
                        'Y3_Calcite_Content': np.mean(y3_values) if y3_values else np.nan,
                        'Alkali_Type': alkali_type,
                        'Y1_Std': np.std(y1_values) if len(y1_values) > 1 else 0,
                        'Y2_Std': np.std(y2_values) if len(y2_values) > 1 else 0,
                        'Y3_Std': np.std(y3_values) if len(y3_values) > 1 else 0,
                        'N_Y1': len(y1_values),
                        'N_Y2': len(y2_values),
                        'N_Y3': len(y3_values)
                    }
                    processed_data.append(data_point)
                    
            except Exception as e:
                continue
        
        df = pd.DataFrame(processed_data)
        
        # Separate Ca and Na data
        self.ca_data = df[df['Alkali_Type'] == 'Ca'].copy()
        self.na_data = df[df['Alkali_Type'] == 'Na'].copy()
        
        print(f"Data loaded - Ca samples: {len(self.ca_data)}, Na samples: {len(self.na_data)}")
        
        # Calculate parameter bounds with 10% extension
        self.calculate_parameter_bounds()
        
        return df
    
    def calculate_parameter_bounds(self):
        """Calculate parameter bounds with 10% extension for prediction space"""
        print("\nCalculating parameter bounds with 10% extension...")
        
        all_data = pd.concat([self.ca_data, self.na_data]) if len(self.ca_data) > 0 and len(self.na_data) > 0 else (self.ca_data if len(self.ca_data) > 0 else self.na_data)
        
        parameters = ['X1_Ca_Conc', 'X2_Time', 'X3_pH', 'X4_CO2_Flow']
        param_names = ['Ca Concentration (ppm)', 'Time (min)', 'pH', 'CO2 Flow (L/M)']
        
        for param, name in zip(parameters, param_names):
            min_val = all_data[param].min()
            max_val = all_data[param].max()
            
            # 10% extension
            range_val = max_val - min_val
            extension = range_val * 0.1
            
            extended_min = min_val - extension
            extended_max = max_val + extension
            
            self.parameter_bounds[param] = {
                'original_min': min_val,
                'original_max': max_val,
                'extended_min': extended_min,
                'extended_max': extended_max,
                'name': name
            }
            
            print(f"{name}:")
            print(f"  Original range: {min_val:.2f} - {max_val:.2f}")
            print(f"  Extended range: {extended_min:.2f} - {extended_max:.2f}")
    
    def train_prediction_models(self, data, system_name):
        """Train comprehensive prediction models for Yi estimation"""
        print(f"\n=== Training Prediction Models for {system_name} ===")
        
        if len(data) < 5:
            print(f"Insufficient data for {system_name}")
            return None
        
        X_features = ['X1_Ca_Conc', 'X2_Time', 'X3_pH', 'X4_CO2_Flow']
        targets = {
            'Y1_Peak_Temp': 'Peak Temperature (°C)',
            'Y2_CO2_Content': 'CO2 Content (%)', 
            'Y3_Calcite_Content': 'Calcite Content (%)'
        }
        
        # ML algorithms optimized for small datasets
        algorithms = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=0.1),  # Lower regularization for small data
            'Lasso_Regression': Lasso(alpha=0.01),  # Lower regularization
            'Random_Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),  # Simpler RF
            'SVR_Linear': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='linear', C=1.0))  # Linear kernel for small data
            ]),
            'SVR_RBF': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=1.0, gamma='scale'))  # Lower C for small data
            ])
        }
        
        system_models = {}
        
        for target, target_desc in targets.items():
            print(f"\nTraining models for {target_desc}:")
            
            # Filter valid data
            valid_data = data[data[target].notna()].copy()
            if len(valid_data) < 5:
                print(f"  Insufficient data ({len(valid_data)} samples)")
                continue
            
            X = valid_data[X_features].values
            y = valid_data[target].values
            
            best_model = None
            best_score = -np.inf
            best_name = ""
            model_results = {}
            
            for alg_name, algorithm in algorithms.items():
                try:
                    # Use smaller CV folds for small datasets
                    cv_folds = min(3, len(valid_data) - 1)  # Use 3-fold CV max
                    if cv_folds < 2:
                        cv_folds = 2
                    
                    # Cross-validation
                    cv_scores = cross_val_score(algorithm, X, y, cv=cv_folds, scoring='r2')
                    
                    # Train on full data
                    algorithm.fit(X, y)
                    y_pred = algorithm.predict(X)
                    
                    # Metrics
                    r2 = r2_score(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    mae = mean_absolute_error(y, y_pred)
                    
                    model_results[alg_name] = {
                        'model': algorithm,
                        'r2_score': r2,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'rmse': rmse,
                        'mae': mae
                    }
                    
                    print(f"  {alg_name}: R² = {r2:.3f}, CV = {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                    
                    # Track best model - prioritize training R² if CV is poor
                    score_to_use = r2 if cv_scores.mean() < 0 else cv_scores.mean()
                    if score_to_use > best_score:
                        best_score = score_to_use
                        best_model = algorithm
                        best_name = alg_name
                        
                except Exception as e:
                    print(f"  {alg_name}: Failed - {str(e)}")
                    continue
            
            if best_model is not None:
                # Calculate feature importance
                feature_importance = self.calculate_feature_importance(best_model, X, y, X_features)
                
                system_models[target] = {
                    'best_model': best_model,
                    'best_name': best_name,
                    'all_results': model_results,
                    'feature_names': X_features,
                    'target_name': target_desc,
                    'feature_importance': feature_importance
                }
                print(f"  → Best model: {best_name} (CV R² = {best_score:.3f})")
                
                # Display feature importance
                print(f"  → Feature importance: {', '.join([f'{feat}: {imp:.3f}' for feat, imp in zip(X_features, feature_importance)])}")
        
        # Store feature importance for 3D plotting
        self.feature_importance[system_name] = system_models
        
        self.best_models[system_name] = system_models
        return system_models
    
    def calculate_feature_importance(self, model, X, y, feature_names):
        """Calculate feature importance for the given model"""
        try:
            # For tree-based models, use built-in feature importance
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            
            # For linear models with coefficients
            elif hasattr(model, 'coef_'):
                # Use absolute values of coefficients as importance
                coef = model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]
                return np.abs(coef) / np.sum(np.abs(coef))  # Normalize
            
            # For pipeline models (like SVR with scaler)
            elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('svr', model), 'coef_'):
                coef = model.named_steps['svr'].coef_[0]
                return np.abs(coef) / np.sum(np.abs(coef))
            
            # Fallback: use permutation importance
            else:
                perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
                return perm_importance.importances_mean
                
        except Exception as e:
            print(f"    Warning: Could not calculate feature importance: {e}")
            # Return equal importance as fallback
            return np.ones(len(feature_names)) / len(feature_names)
    
    def create_3d_surface_plots(self, system_name):
        """Create 3D surface plots based on the two most dominant Xi descriptors"""
        print(f"\nCreating 3D surface plots for {system_name}...")
        
        if system_name not in self.best_models:
            print(f"No models available for {system_name}")
            return
        
        system_models = self.best_models[system_name]
        data = self.ca_data if system_name == "Calcium" else self.na_data
        
        # Parameter names for plotting
        param_names = {
            'X1_Ca_Conc': 'Ca Concentration (ppm)',
            'X2_Time': 'Time (min)',
            'X3_pH': 'pH',
            'X4_CO2_Flow': 'CO2 Flow (L/M)'
        }
        
        n_targets = len(system_models)
        fig = plt.figure(figsize=(18, 6*n_targets))
        
        plot_idx = 1
        
        for target, model_info in system_models.items():
            print(f"  Creating 3D plot for {model_info['target_name']}...")
            
            # Get feature importance and find top 2 features
            feature_importance = model_info['feature_importance']
            feature_names = model_info['feature_names']
            
            # Sort features by importance
            importance_pairs = list(zip(feature_names, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Get top 2 most important features
            top_feature_1 = importance_pairs[0][0]
            top_feature_2 = importance_pairs[1][0]
            
            print(f"    Most dominant features: {top_feature_1} ({importance_pairs[0][1]:.3f}), {top_feature_2} ({importance_pairs[1][1]:.3f})")
            
            # Get data for valid predictions
            valid_data = data[data[target].notna()].copy()
            if len(valid_data) < 3:
                print(f"    Insufficient data for 3D plot")
                continue
            
            # Create parameter ranges for the two most important features
            x1_values = valid_data[top_feature_1].values
            x2_values = valid_data[top_feature_2].values
            y_values = valid_data[target].values
            
            # Extend ranges by 5% for better visualization
            x1_min, x1_max = x1_values.min(), x1_values.max()
            x2_min, x2_max = x2_values.min(), x2_values.max()
            
            x1_range = x1_max - x1_min
            x2_range = x2_max - x2_min
            
            x1_extended_min = x1_min - 0.05 * x1_range
            x1_extended_max = x1_max + 0.05 * x1_range
            x2_extended_min = x2_min - 0.05 * x2_range
            x2_extended_max = x2_max + 0.05 * x2_range
            
            # Create mesh grid for surface
            x1_mesh = np.linspace(x1_extended_min, x1_extended_max, 30)
            x2_mesh = np.linspace(x2_extended_min, x2_extended_max, 30)
            X1_grid, X2_grid = np.meshgrid(x1_mesh, x2_mesh)
            
            # Create prediction inputs - use median values for other features
            other_features = [f for f in feature_names if f not in [top_feature_1, top_feature_2]]
            median_values = {}
            for feat in other_features:
                median_values[feat] = valid_data[feat].median()
            
            # Generate predictions for the grid
            Z_pred = np.zeros_like(X1_grid)
            
            for i in range(X1_grid.shape[0]):
                for j in range(X1_grid.shape[1]):
                    # Create input vector
                    input_dict = {
                        top_feature_1: X1_grid[i, j],
                        top_feature_2: X2_grid[i, j]
                    }
                    input_dict.update(median_values)
                    
                    # Order according to original feature order
                    input_vector = np.array([[input_dict[feat] for feat in feature_names]])
                    
                    try:
                        Z_pred[i, j] = model_info['best_model'].predict(input_vector)[0]
                    except:
                        Z_pred[i, j] = np.nan
            
            # Create 3D subplot
            ax = fig.add_subplot(n_targets, 1, plot_idx, projection='3d')
            
            # Plot surface
            surf = ax.plot_surface(X1_grid, X2_grid, Z_pred, 
                                 cmap='viridis', alpha=0.7, 
                                 linewidth=0, antialiased=True)
            
            # Plot actual data points
            ax.scatter(x1_values, x2_values, y_values, 
                      c='red', s=50, alpha=0.8, label='Actual Data')
            
            # Labels and title
            ax.set_xlabel(f'{top_feature_1}\n{param_names.get(top_feature_1, top_feature_1)}')
            ax.set_ylabel(f'{top_feature_2}\n{param_names.get(top_feature_2, top_feature_2)}')
            ax.set_zlabel(f'{model_info["target_name"]}')
            ax.set_title(f'{system_name} - {model_info["target_name"]}\n'
                        f'Dominant Features: {top_feature_1} & {top_feature_2}\n'
                        f'Model: {model_info["best_name"]}')
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
            ax.legend()
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'H:/FreeLancer/Elmira/{system_name}_3D_surfaces.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_importance_plots(self):
        """Create feature importance comparison plots"""
        print("\nCreating feature importance analysis...")
        
        # Collect all feature importance data
        importance_data = []
        
        for system_name, models in self.best_models.items():
            for target, model_info in models.items():
                feature_names = model_info['feature_names']
                feature_importance = model_info['feature_importance']
                
                for feat, imp in zip(feature_names, feature_importance):
                    importance_data.append({
                        'System': system_name,
                        'Target': model_info['target_name'],
                        'Feature': feat,
                        'Importance': imp,
                        'Model': model_info['best_name']
                    })
        
        if not importance_data:
            print("No feature importance data available")
            return
        
        df_importance = pd.DataFrame(importance_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Importance Analysis - Xi Dominance for Yi Prediction', fontsize=16)
        
        # Plot 1: Heatmap of feature importance by system and target
        pivot_data = df_importance.pivot_table(
            values='Importance', 
            index=['System', 'Target'], 
            columns='Feature', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, cmap='viridis', ax=axes[0,0])
        axes[0,0].set_title('Feature Importance Heatmap')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Bar plot for Calcium system
        ca_data = df_importance[df_importance['System'] == 'Calcium']
        if len(ca_data) > 0:
            ca_pivot = ca_data.pivot(index='Feature', columns='Target', values='Importance')
            ca_pivot.plot(kind='bar', ax=axes[0,1])
            axes[0,1].set_title('Calcium System - Feature Importance')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Bar plot for Sodium system
        na_data = df_importance[df_importance['System'] == 'Sodium']
        if len(na_data) > 0:
            na_pivot = na_data.pivot(index='Feature', columns='Target', values='Importance')
            na_pivot.plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('Sodium System - Feature Importance')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: Overall feature ranking
        overall_importance = df_importance.groupby('Feature')['Importance'].mean().sort_values(ascending=True)
        overall_importance.plot(kind='barh', ax=axes[1,1])
        axes[1,1].set_title('Overall Feature Importance Ranking')
        axes[1,1].set_xlabel('Average Importance')
        
        plt.tight_layout()
        plt.savefig('H:/FreeLancer/Elmira/Feature_Importance_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print feature importance summary
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE SUMMARY")
        print("="*60)
        
        for system_name, models in self.best_models.items():
            print(f"\n{system_name.upper()} SYSTEM:")
            print("-" * 30)
            
            for target, model_info in models.items():
                print(f"\n{model_info['target_name']}:")
                feature_names = model_info['feature_names']
                feature_importance = model_info['feature_importance']
                
                # Sort by importance
                sorted_features = sorted(zip(feature_names, feature_importance), 
                                       key=lambda x: x[1], reverse=True)
                
                for i, (feat, imp) in enumerate(sorted_features, 1):
                    param_name = {
                        'X1_Ca_Conc': 'Ca Concentration',
                        'X2_Time': 'Time',
                        'X3_pH': 'pH',
                        'X4_CO2_Flow': 'CO2 Flow'
                    }.get(feat, feat)
                    
                    print(f"  {i}. {param_name}: {imp:.4f} ({'★★' if i <= 2 else '★' if i == 3 else ''})")
        
        return df_importance
    
    def generate_prediction_grid(self, system_name, grid_resolution=20):
        """Generate comprehensive prediction grid across parameter space"""
        print(f"\nGenerating prediction grid for {system_name}...")
        
        if system_name not in self.best_models:
            print(f"No models available for {system_name}")
            return None
        
        # Create parameter grid
        x1_range = np.linspace(
            self.parameter_bounds['X1_Ca_Conc']['extended_min'],
            self.parameter_bounds['X1_Ca_Conc']['extended_max'],
            grid_resolution
        )
        x2_range = np.linspace(
            self.parameter_bounds['X2_Time']['extended_min'],
            self.parameter_bounds['X2_Time']['extended_max'],
            grid_resolution
        )
        x3_range = np.linspace(
            self.parameter_bounds['X3_pH']['extended_min'],
            self.parameter_bounds['X3_pH']['extended_max'],
            grid_resolution
        )
        x4_range = np.linspace(
            self.parameter_bounds['X4_CO2_Flow']['extended_min'],
            self.parameter_bounds['X4_CO2_Flow']['extended_max'],
            grid_resolution
        )
        
        # Create meshgrid for all combinations
        X1, X2, X3, X4 = np.meshgrid(x1_range, x2_range, x3_range, x4_range, indexing='ij')
        grid_shape = X1.shape
        
        # Flatten for prediction
        X_grid = np.column_stack([
            X1.flatten(),
            X2.flatten(),
            X3.flatten(),
            X4.flatten()
        ])
        
        predictions = {}
        
        for target, model_info in self.best_models[system_name].items():
            print(f"  Predicting {target}...")
            
            try:
                y_pred_flat = model_info['best_model'].predict(X_grid)
                y_pred_grid = y_pred_flat.reshape(grid_shape)
                
                predictions[target] = {
                    'values': y_pred_grid,
                    'model_name': model_info['best_name'],
                    'target_desc': model_info['target_name']
                }
                
            except Exception as e:
                print(f"    Failed to predict {target}: {str(e)}")
                continue
        
        grid_data = {
            'X1_grid': X1,
            'X2_grid': X2,
            'X3_grid': X3,
            'X4_grid': X4,
            'predictions': predictions,
            'parameter_ranges': {
                'X1_range': x1_range,
                'X2_range': x2_range,
                'X3_range': x3_range,
                'X4_range': x4_range
            }
        }
        
        self.prediction_grids[system_name] = grid_data
        print(f"  Grid generated: {grid_resolution}^4 = {grid_resolution**4:,} prediction points")
        
        return grid_data
    
    def create_prediction_visualization(self, system_name):
        """Create comprehensive visualization of predictions"""
        if system_name not in self.prediction_grids:
            print(f"No prediction grid available for {system_name}")
            return
        
        grid_data = self.prediction_grids[system_name]
        predictions = grid_data['predictions']
        
        if not predictions:
            print(f"No predictions available for {system_name}")
            return
        
        # Create 2D slices at median values of other parameters
        target_list = list(predictions.keys())
        n_targets = len(target_list)
        
        fig, axes = plt.subplots(n_targets, 3, figsize=(18, 6*n_targets))
        if n_targets == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{system_name} - Yi Predictions across Xi Parameter Space', fontsize=16)
        
        # Parameter names for labels
        param_labels = {
            'X1': 'Ca Conc (ppm)',
            'X2': 'Time (min)', 
            'X3': 'pH',
            'X4': 'CO2 Flow (L/M)'
        }
        
        # Parameter combinations for 2D visualization
        param_combinations = [
            ('X1', 'X2', 2, 3),  # X1 vs X2, fix X3, X4
            ('X1', 'X3', 1, 3),  # X1 vs X3, fix X2, X4
            ('X2', 'X3', 0, 3),  # X2 vs X3, fix X1, X4
        ]
        
        for i, target in enumerate(target_list):
            y_pred = predictions[target]['values']
            target_desc = predictions[target]['target_desc']
            
            for j, (x_param, y_param, fix1_idx, fix2_idx) in enumerate(param_combinations):
                ax = axes[i, j] if n_targets > 1 else axes[j]
                
                # Get median indices for fixed parameters
                mid_idx1 = y_pred.shape[fix1_idx] // 2
                mid_idx2 = y_pred.shape[fix2_idx] // 2
                
                # Extract 2D slice based on parameter combination
                if x_param == 'X1' and y_param == 'X2':
                    slice_data = y_pred[:, :, mid_idx1, mid_idx2]
                    x_range = grid_data['parameter_ranges']['X1_range']
                    y_range = grid_data['parameter_ranges']['X2_range']
                elif x_param == 'X1' and y_param == 'X3':
                    slice_data = y_pred[:, mid_idx1, :, mid_idx2]
                    x_range = grid_data['parameter_ranges']['X1_range']
                    y_range = grid_data['parameter_ranges']['X3_range']
                elif x_param == 'X2' and y_param == 'X3':
                    slice_data = y_pred[mid_idx1, :, :, mid_idx2]
                    x_range = grid_data['parameter_ranges']['X2_range']
                    y_range = grid_data['parameter_ranges']['X3_range']
                
                # Create contour plot
                try:
                    X_mesh, Y_mesh = np.meshgrid(x_range, y_range, indexing='ij')
                    contour = ax.contourf(X_mesh, Y_mesh, slice_data, levels=20, cmap='viridis')
                    ax.contour(X_mesh, Y_mesh, slice_data, levels=10, colors='white', alpha=0.4, linewidths=0.5)
                    
                    ax.set_xlabel(f'{x_param} - {param_labels[x_param]}')
                    ax.set_ylabel(f'{y_param} - {param_labels[y_param]}')
                    
                    if i == 0:
                        ax.set_title(f'{x_param} vs {y_param}')
                    
                    if j == 0:
                        ax.text(-0.1, 0.5, target_desc, rotation=90, 
                               transform=ax.transAxes, ha='center', va='center')
                    
                    plt.colorbar(contour, ax=ax, shrink=0.8)
                    
                except Exception as e:
                    print(f"Error creating plot for {target} {x_param} vs {y_param}: {e}")
                    ax.text(0.5, 0.5, f'Plot Error\n{target}\n{x_param} vs {y_param}', 
                           transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(f'H:/FreeLancer/Elmira/{system_name}_prediction_maps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_for_custom_conditions(self, system_name, x1, x2, x3, x4):
        """Predict Yi values for specific Xi conditions"""
        if system_name not in self.best_models:
            print(f"No models available for {system_name}")
            return None
        
        print(f"\nPredicting for {system_name} with conditions:")
        print(f"  X1 (Ca Conc): {x1} ppm")
        print(f"  X2 (Time): {x2} min")
        print(f"  X3 (pH): {x3}")
        print(f"  X4 (CO2 Flow): {x4} L/M")
        
        X_input = np.array([[x1, x2, x3, x4]])
        predictions = {}
        
        for target, model_info in self.best_models[system_name].items():
            try:
                y_pred = model_info['best_model'].predict(X_input)[0]
                predictions[target] = {
                    'value': y_pred,
                    'target_desc': model_info['target_name'],
                    'model_used': model_info['best_name']
                }
                print(f"  {model_info['target_name']}: {y_pred:.3f} (using {model_info['best_name']})")
            except Exception as e:
                print(f"  Failed to predict {target}: {str(e)}")
        
        return predictions
    
    def save_comprehensive_results(self):
        """Save all results including prediction grids"""
        print("\nSaving comprehensive results...")
        
        with pd.ExcelWriter('H:/FreeLancer/Elmira/Yi_Prediction_Analysis.xlsx', engine='openpyxl') as writer:
            # Save original data
            if len(self.ca_data) > 0:
                self.ca_data.to_excel(writer, sheet_name='Ca_Data', index=False)
            if len(self.na_data) > 0:
                self.na_data.to_excel(writer, sheet_name='Na_Data', index=False)
            
            # Save parameter bounds
            bounds_data = []
            for param, bounds in self.parameter_bounds.items():
                bounds_data.append({
                    'Parameter': param,
                    'Parameter_Name': bounds['name'],
                    'Original_Min': bounds['original_min'],
                    'Original_Max': bounds['original_max'],
                    'Extended_Min': bounds['extended_min'],
                    'Extended_Max': bounds['extended_max']
                })
            pd.DataFrame(bounds_data).to_excel(writer, sheet_name='Parameter_Bounds', index=False)
            
            # Save model performance
            model_performance = []
            for system, models in self.best_models.items():
                for target, model_info in models.items():
                    best_result = model_info['all_results'][model_info['best_name']]
                    model_performance.append({
                        'System': system,
                        'Target': target,
                        'Target_Description': model_info['target_name'],
                        'Best_Model': model_info['best_name'],
                        'R2_Score': best_result['r2_score'],
                        'CV_Mean': best_result['cv_mean'],
                        'CV_Std': best_result['cv_std'],
                        'RMSE': best_result['rmse'],
                        'MAE': best_result['mae']
                    })
            
            if model_performance:
                pd.DataFrame(model_performance).to_excel(writer, sheet_name='Model_Performance', index=False)
            
            # Save sample predictions at key points
            sample_predictions = []
            for system in self.best_models.keys():
                # Predict at min, median, max conditions
                bounds = self.parameter_bounds
                conditions = [
                    ('min', bounds['X1_Ca_Conc']['extended_min'], bounds['X2_Time']['extended_min'], 
                     bounds['X3_pH']['extended_min'], bounds['X4_CO2_Flow']['extended_min']),
                    ('median', (bounds['X1_Ca_Conc']['extended_min'] + bounds['X1_Ca_Conc']['extended_max'])/2,
                     (bounds['X2_Time']['extended_min'] + bounds['X2_Time']['extended_max'])/2,
                     (bounds['X3_pH']['extended_min'] + bounds['X3_pH']['extended_max'])/2,
                     (bounds['X4_CO2_Flow']['extended_min'] + bounds['X4_CO2_Flow']['extended_max'])/2),
                    ('max', bounds['X1_Ca_Conc']['extended_max'], bounds['X2_Time']['extended_max'],
                     bounds['X3_pH']['extended_max'], bounds['X4_CO2_Flow']['extended_max'])
                ]
                
                for condition_name, x1, x2, x3, x4 in conditions:
                    preds = self.predict_for_custom_conditions(system, x1, x2, x3, x4)
                    if preds:
                        for target, pred_info in preds.items():
                            sample_predictions.append({
                                'System': system,
                                'Condition': condition_name,
                                'X1_Ca_Conc': x1,
                                'X2_Time': x2,
                                'X3_pH': x3,
                                'X4_CO2_Flow': x4,
                                'Target': target,
                                'Predicted_Value': pred_info['value'],
                                'Model_Used': pred_info['model_used']
                            })
            
            if sample_predictions:
                pd.DataFrame(sample_predictions).to_excel(writer, sheet_name='Sample_Predictions', index=False)
        
        print("Results saved to H:/FreeLancer/Elmira/Yi_Prediction_Analysis.xlsx")
    
    def run_complete_analysis(self):
        """Run the complete prediction analysis"""
        print("Starting Yi Prediction Analysis")
        print("="*50)
        
        # Load and process data
        self.load_and_process_data()
        
        # Train models for each system
        if len(self.ca_data) > 0:
            print(f"\nTraining models for Calcium system ({len(self.ca_data)} samples)...")
            try:
                self.train_prediction_models(self.ca_data, "Calcium")
                self.generate_prediction_grid("Calcium", grid_resolution=10)  # Smaller grid for speed
                self.create_prediction_visualization("Calcium")
                self.create_3d_surface_plots("Calcium")  # New 3D plots
            except Exception as e:
                print(f"Error processing Calcium system: {e}")
        
        if len(self.na_data) > 0:
            print(f"\nTraining models for Sodium system ({len(self.na_data)} samples)...")
            try:
                self.train_prediction_models(self.na_data, "Sodium")
                self.generate_prediction_grid("Sodium", grid_resolution=10)  # Smaller grid for speed
                self.create_prediction_visualization("Sodium")
                self.create_3d_surface_plots("Sodium")  # New 3D plots
            except Exception as e:
                print(f"Error processing Sodium system: {e}")
        
        # Create feature importance analysis
        try:
            self.create_feature_importance_plots()
        except Exception as e:
            print(f"Error creating feature importance plots: {e}")
        
        # Example predictions
        print("\n" + "="*50)
        print("EXAMPLE PREDICTIONS")
        print("="*50)
        
        # Example conditions
        example_conditions = [
            (3000, 25, 9.0, 0.15),  # Mid-range conditions
            (4000, 35, 8.5, 0.2),   # High Ca, high time, low pH, high flow
            (2500, 30, 9.5, 0.1),   # Low Ca, mid time, high pH, low flow
        ]
        
        for i, (x1, x2, x3, x4) in enumerate(example_conditions, 1):
            print(f"\nExample {i}:")
            for system in self.best_models.keys():
                self.predict_for_custom_conditions(system, x1, x2, x3, x4)
        
        # Save all results
        self.save_comprehensive_results()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("Generated files:")
        print("- Yi_Prediction_Analysis.xlsx (comprehensive results)")
        print("- Calcium_prediction_maps.png (Ca prediction visualizations)")
        print("- Sodium_prediction_maps.png (Na prediction visualizations)")
        print("\nYou can now predict Y1, Y2, Y3 for any Xi combination within the bounds!")


def main():
    """Main function"""
    file_path = r"H:\FreeLancer\Elmira\DataSpreadsheet.xlsx"
    
    analyzer = YiPredictionAnalyzer(file_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
