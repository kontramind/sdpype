"""Core metrics information and utilities"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional, Dict, Any

console = Console()

# Available evaluation metrics registry
# Available evaluation metrics registry
AVAILABLE_METRICS = {
    "statistical": {
        "alpha_precision": {
            "description": "Alpha Precision metric with optimally-corrected and naive variants",
            "library": "synthcity",
            "parameters": {},  # No configurable parameters
            "outputs": ["delta_precision_alpha_OC", "delta_coverage_beta_OC", "authenticity_OC", 
                       "delta_precision_alpha_naive", "delta_coverage_beta_naive", "authenticity_naive"],
            "reference": "Alaa et al. 2022 - How faithful is your synthetic data?"
        },
        "prdc_score": {
            "description": "Precision, Recall, Density, and Coverage for manifold comparison", 
            "library": "synthcity",
            "parameters": {
                "nearest_k": {
                    "type": "int",
                    "default": 5,
                    "description": "Number of nearest neighbors for distance calculations"
                }
            },
            "outputs": ["precision", "recall", "density", "coverage"],
            "reference": "Kynkäänniemi et al. 2019 - Improved precision and recall metric"
        },
        "wasserstein_distance": {
            "description": "Wasserstein distance (Earth Mover's Distance) between distributions",
            "library": "synthcity",
            "parameters": {},  # No configurable parameters
            "outputs": ["joint_distance"],
            "reference": "Optimal transport distance - lower values indicate more similar distributions",
            "direction": "minimize",
        },
        "maximum_mean_discrepancy": {
            "description": "Maximum Mean Discrepancy (MMD) between distributions using kernel methods",
            "library": "synthcity",
            "parameters": {
                "kernel": {
                    "type": "str",
                    "default": "rbf",
                    "options": ["rbf", "linear", "polynomial"],
                    "description": "Kernel function for MMD computation"
                }
            },
            "outputs": ["joint_distance"],
            "reference": "Kernel two-sample test - lower values indicate more similar distributions",
            "direction": "minimize",
            "interpretation": {
                "excellent": "< 0.001",
                "good": "< 0.01",
                "moderate": "< 0.1"
            }
        },
        "jensenshannon_synthcity": {
            "description": "Jensen-Shannon Distance between distributions (Synthcity implementation)",
            "library": "synthcity",
            "parameters": {
                "normalize": {
                    "type": "bool",
                    "default": True,
                    "description": "Whether to normalize distributions before computing JSD"
                },
                "n_histogram_bins": {
                    "type": "int",
                    "default": 10,
                    "description": "Number of histogram bins for continuous variables"
                }
            },
            "outputs": ["distance_score"],
            "reference": "Symmetric measure of distance between probability distributions",
            "direction": "minimize",
            "interpretation": {
                "excellent": "< 0.01",
                "good": "< 0.05",
                "moderate": "< 0.1"
            }
        },
        "jensenshannon_nannyml": {
            "description": "Jensen-Shannon Distance between distributions (NannyML implementation with adaptive binning)",
            "library": "nannyml",
            "parameters": {},  # Uses adaptive Doane's formula for binning
            "outputs": ["distance_score"],
            "reference": "Production-grade implementation with Doane's formula binning",
            "direction": "minimize",
            "interpretation": {
                "excellent": "< 0.01",
                "good": "< 0.05",
                "moderate": "< 0.1"
            }
        },
        "jensenshannon_syndat": {
            "description": "Jensen-Shannon Distance between distributions (SYNDAT implementation)",
            "library": "syndat",
            "parameters": {
                "n_unique_threshold": {
                    "type": "int",
                    "default": 10,
                    "description": "Max unique values for ordinal treatment"
                }
            },
            "outputs": ["distance_score"],
            "reference": "Symmetric measure of distance between probability distributions",
            "direction": "minimize",
            "interpretation": {
                "excellent": "< 0.01",
                "good": "< 0.05",
                "moderate": "< 0.1"
            }
        },
        "new_row_synthesis": {
            "description": "Measures whether each synthetic row is new or matches real data exactly",
            "library": "sdmetrics",
            "parameters": {
                "numerical_match_tolerance": {
                    "type": "float",
                    "default": 0.01,
                    "description": "Tolerance for numerical value matching (0.01 = 1%)"
                },
                "synthetic_sample_size": {
                    "type": "int",
                    "default": None,
                    "description": "Number of synthetic rows to sample (None = use all rows)"
                }
            },
            "outputs": ["score", "num_new_rows", "num_matched_rows"],
            "reference": "SDMetrics library - Single table evaluation metric"
        },
        "ks_complement": {
            "description": "Kolmogorov-Smirnov test for distribution similarity across numerical/datetime columns",
            "library": "sdmetrics",
            "parameters": {
                "target_columns": {
                    "type": "list",
                    "default": None,
                    "description": "Specific columns to evaluate (None = all numerical/datetime columns)"
                }
            },
            "outputs": ["aggregate_score", "column_scores", "compatible_columns"],
            "reference": "SDMetrics library - Single column statistical metric using KS test"
        },
        "tv_complement": {
            "description": "Total Variation Distance for categorical distribution similarity across categorical/boolean columns",
            "library": "sdmetrics",
            "parameters": {
                "target_columns": {
                    "type": "list",
                    "default": None,
                    "description": "Specific columns to evaluate (None = all categorical/boolean columns)"
                }
            },
            "outputs": ["aggregate_score", "column_scores", "compatible_columns"],
            "reference": "SDMetrics library - Single column statistical metric using Total Variation Distance"
        }
    },
    "diagnostic": {
        "table_structure": {
            "description": "Measures whether synthetic data captures the same table structure (column names and data types) as real data",
            "library": "sdmetrics",
            "parameters": {},  # No configurable parameters
            "outputs": ["score"],
            "reference": "SDMetrics library - Single table diagnostic metric for structure validation"
        },
        "semantic_structure": {
            "description": "Measures whether synthetic data captures the same semantic column types (sdtypes) as real data, more forgiving than strict dtype matching",
            "library": "sdpype",
            "parameters": {},  # No configurable parameters
            "outputs": ["score", "summary", "column_details", "comparison_table"],
            "reference": "Uses SDV metadata sdtypes (numerical, categorical, datetime, etc.) instead of pandas dtypes (int64, float64, object, etc.) for semantic type comparison"
        },
        "boundary_adherence": {
            "description": "Measures whether synthetic column values respect the min/max boundaries of real data",
            "library": "sdmetrics",
            "parameters": {
                "target_columns": {
                    "type": "list",
                    "default": None,
                    "description": "Specific columns to evaluate (None = all numerical/datetime columns)"
                }
            },
            "outputs": ["aggregate_score", "column_scores", "compatible_columns"],
            "reference": "SDMetrics library - Single column diagnostic metric for boundary validation"
        },
        "category_adherence": {
            "description": "Measures whether synthetic categorical data adheres to the same category values as real data (no new categories)",
            "library": "sdmetrics",
            "parameters": {
                "target_columns": {
                    "type": "list",
                    "default": None,
                    "description": "Specific columns to evaluate (None = all categorical/boolean columns)"
                }
            },
            "outputs": ["aggregate_score", "column_scores", "compatible_columns"],
            "reference": "SDMetrics library - Single column diagnostic metric for category validation"
        }
    }
}


def show_available_metrics(show_params: bool = False, metric_type: Optional[str] = None):
    """Show available evaluation metrics"""

    console.print("📊 Available Evaluation Metrics\n", style="bold blue")

    # Create simple table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Library", style="green", no_wrap=True)
    table.add_column("Category", style="blue")
    table.add_column("Description", style="white")

    for category_name, metrics in AVAILABLE_METRICS.items():
        for metric_name, info in metrics.items():
            desc = info['description'][:60] + "..." if len(info['description']) > 60 else info['description']
            table.add_row(metric_name, info.get('library', 'unknown'), category_name, desc)

    console.print(table)

    _show_usage_examples()

def _show_usage_examples():
    """Show usage examples for metrics"""

    console.print("\n💡 Usage Examples:", style="bold")
    examples = [
        "sdpype metrics info synthcity/alpha_precision  # Show metric details",
        "sdpype metrics info synthcity/prdc_score       # Show parameters and examples"
    ]

    for example in examples:
        console.print(f"  {example}")

def show_metric_details(metric_name: str):
    """Show detailed information about a specific metric"""  

    # Require library/metric format
    if '/' not in metric_name:
        console.print(f"❌ Please specify the full path: library/metric", style="red")
        console.print("\nAvailable metrics:")
        for cat_name, metrics in AVAILABLE_METRICS.items():
            for name, info in metrics.items():
                lib = info.get('library', 'unknown')
                console.print(f"  • {lib}/{name}")
        return

    # Parse library/metric format
    library, metric = metric_name.split('/', 1)

    # Find the metric
    metric_info = None
    category = None

    for cat_name, metrics in AVAILABLE_METRICS.items():
        if metric in metrics:
            candidate_info = metrics[metric]
            candidate_library = candidate_info.get('library', 'unknown')

            # Library must match
            if candidate_library != library:
                 continue

            metric_info = candidate_info
            category = cat_name
            break

    if not metric_info:
        console.print(f"❌ Metric '{metric}' not found in library '{library}'", style="red")
        console.print("\nAvailable metrics:")
        for cat_name, metrics in AVAILABLE_METRICS.items():
            for name, info in metrics.items():
                lib = info.get('library', 'unknown')
                console.print(f"  • {lib}/{name}")
        return

    # Show detailed information
    console.print(f"📊 {metric} Details\n", style="bold blue")

    console.print(f"[bold]Library:[/bold] {library}")
    console.print(f"[bold]Category:[/bold] {category}")
    console.print(f"[bold]Description:[/bold] {metric_info['description']}")

    # Parameters section
    console.print(f"\n[bold]Parameters:[/bold]")
    if metric_info.get("parameters"):
        for param_name, param_info in metric_info["parameters"].items():
            console.print(f"  • [cyan]{param_name}[/cyan] ({param_info['type']})")
            console.print(f"    Default: {param_info.get('default', 'none')}")
            console.print(f"    {param_info['description']}")
    else:
        console.print("  No configurable parameters")

    # Outputs section
    console.print(f"\n[bold]Output Scores:[/bold]")
    for output in metric_info['outputs']:
        console.print(f"  • {output}")

    if metric_info.get("reference"):
        console.print(f"\n[bold]Reference:[/bold] {metric_info['reference']}")

    # Show configuration example
    console.print(f"\n[bold]Configuration Example:[/bold]")
    console.print("```yaml")
    console.print("evaluation:")
    console.print("  statistical_similarity:")
    console.print("    metrics:")
    console.print(f"      - name: {metric_name}")
    if metric_info.get("parameters"):
        console.print("        parameters:")
        for param_name, param_info in metric_info["parameters"].items():
            default_val = param_info.get("default")
            console.print(f"          {param_name}: {default_val}")
    else:
        console.print("        parameters: {}")
    console.print("```")
