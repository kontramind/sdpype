"""
Detection-based evaluation script for SDPype - Real vs Synthetic Classification
"""

import json
from pathlib import Path
from typing import Dict, Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sdv.metadata import SingleTableMetadata
from sdpype.evaluation.detection import evaluate_detection_metrics, generate_detection_report
from sdpype.encoding import load_encoding_config

console = Console()


def _get_config_hash() -> str:
    """Get config hash from temporary file created during pipeline execution"""
    try:
        if Path('.sdpype_config_hash').exists():
            with open('.sdpype_config_hash', 'r', encoding='utf-8') as f:
                return f.read().strip()
        return "nohash"
    except Exception:
        return "nohash"


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Run detection-based evaluation between original and synthetic data"""

    # Check if detection evaluation is configured
    methods_config = cfg.get("evaluation", {}).get("detection_evaluation", {}).get("methods", [])

    if not methods_config:
        print("🔍 No detection methods configured, skipping detection evaluation")
        return

    print(f"🔍 Starting detection evaluation with {len(methods_config)} methods...")
    print(f"Experiment seed: {cfg.experiment.seed}")

    config_hash = _get_config_hash()
    base_name = f"{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}"

    # Load datasets for detection evaluation
    # All detection metrics are from synthcity and need encoded (all-numeric) data
    metadata_path = cfg.data.metadata_file
    reference_encoded_path = f"experiments/data/encoded/reference_{base_name}.csv"
    synthetic_encoded_path = f"experiments/data/encoded/synthetic_{base_name}.csv"

    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not Path(reference_encoded_path).exists():
        raise FileNotFoundError(f"Encoded reference data not found: {reference_encoded_path}")
    if not Path(synthetic_encoded_path).exists():
        raise FileNotFoundError(f"Encoded synthetic data not found: {synthetic_encoded_path}")

    print(f"📊 Loading encoded reference data: {reference_encoded_path}")
    print(f"📊 Loading encoded synthetic data: {synthetic_encoded_path}")
    print(f"ℹ️  Detection metrics use encoded data (all-numeric for synthcity)")

    reference_data = pd.read_csv(reference_encoded_path)
    synthetic_data = pd.read_csv(synthetic_encoded_path)
    metadata = SingleTableMetadata.load_from_json(metadata_path)

    # Load encoding config (for determining numeric columns and excluding IDs)
    encoding_config = None
    if hasattr(cfg, 'encoding') and cfg.encoding.get('config_file'):
        encoding_config_path = Path(cfg.encoding.config_file)
        if encoding_config_path.exists():
            print(f"📋 Loading encoding config: {encoding_config_path}")
            encoding_config = load_encoding_config(encoding_config_path)
            print(f"✓ Encoding config loaded - IDs will be excluded from detection")
        else:
            print(f"⚠️  Warning: Encoding config not found at {encoding_config_path}")
            print(f"   Detection will use fallback column detection (ID leakage possible!)")
    else:
        print(f"⚠️  Warning: No encoding config specified in params.yaml")
        print(f"   Detection will use fallback column detection (ID leakage possible!)")

    # Run detection evaluation
    print("🔄 Running detection metrics analysis...")

    # Get common parameters from config
    common_params = cfg.get("evaluation", {}).get("detection_evaluation", {}).get("common_params", {
        "n_folds": 5,
        "random_state": 42,
        "reduction": "mean",
    })

    detection_results = evaluate_detection_metrics(
        reference_data, synthetic_data, metadata, methods_config, common_params,
        cfg.experiment.name, encoding_config
    )

    # Handle evaluation errors
    if "error" in detection_results:
        console.print(f"❌ Detection evaluation failed: {detection_results['error']}", style="bold red")
        console.print("💡 Check data format and synthcity installation", style="yellow")
        return

    # Save results
    results_file = f"experiments/metrics/detection_evaluation_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.json"
    
    print(f"💾 Saving detection results to: {results_file}")
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)

    # Ensure JSON serializable before saving
    from sdpype.evaluation.detection import ensure_json_serializable
    detection_results = ensure_json_serializable(detection_results)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, indent=2, ensure_ascii=False)

    # Generate human-readable report
    report_file = f"experiments/metrics/detection_report_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.txt"
    report_content = generate_detection_report(detection_results)
    
    print(f"📋 Generating detection report: {report_file}")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    # Display detection results table
    _display_detection_tables(detection_results)

    print("✅ Detection evaluation completed successfully!")


def _display_detection_tables(results: Dict[str, Any]):
   """Display detection results in rich tables"""
   
   console.print("\n🔍 Detection-Based Quality Assessment:", style="bold cyan")
   
   individual_scores = results.get("individual_scores", {})
   
   # Create main detection results table
   detection_table = Table(
       title="📊 Synthcity Detection Performance",
       show_header=True, 
       header_style="bold blue"
   )
   detection_table.add_column("Method", style="cyan", no_wrap=True)
   detection_table.add_column("AUC Score", style="bright_green", justify="right")
   detection_table.add_column("Quality Assessment", style="yellow")
   detection_table.add_column("Status", style="green")
   
   # Add individual method results
   for method_name, method_result in individual_scores.items():
       method_display = method_name.replace('_', ' ').title()
       
       if method_result["status"] == "success":
           auc = method_result["auc_score"]
           
           # Quality assessment based on AUC
           if auc <= 0.55:
               quality = "🟢 Excellent"
               quality_style = "bright_green"
           elif auc <= 0.65:
               quality = "🟡 Good"  
               quality_style = "yellow"
           elif auc <= 0.75:
               quality = "🟠 Fair"
               quality_style = "bright_yellow"
           else:
               quality = "🔴 Poor"
               quality_style = "bright_red"
           
           detection_table.add_row(
               method_display,
               f"{auc:.3f}",
               quality,
               "✅ Success"
           )
       else:
           detection_table.add_row(
               method_display,
               "N/A",
               "❌ Failed",
               f"Error: {method_result.get('error_message', 'Unknown')[:30]}..."
           )
   
   console.print(detection_table)
   
   # Ensemble results table
   ensemble_score = results.get("ensemble_score")
   if ensemble_score is not None:
       ensemble_table = Table(
           title="🎯 Ensemble Assessment", 
           show_header=True, 
           header_style="bold magenta"
       )
       ensemble_table.add_column("Metric", style="cyan")
       ensemble_table.add_column("Score", style="bright_green", justify="right")
       ensemble_table.add_column("Overall Quality", style="yellow")
       
       # Overall quality assessment
       if ensemble_score <= 0.55:
           overall_quality = "🟢 Excellent - Very hard to detect"
       elif ensemble_score <= 0.65:
           overall_quality = "🟡 Good - Moderately hard to detect"
       elif ensemble_score <= 0.75:
           overall_quality = "🟠 Fair - Somewhat detectable"
       else:
           overall_quality = "🔴 Poor - Easily detectable"
       
       ensemble_table.add_row(
           "Mean AUC Score",
           f"{ensemble_score:.3f}",
           overall_quality
       )
       
       console.print(ensemble_table)
   else:
       console.print("❌ No ensemble score available", style="bold red")
   
   # Add interpretation guide
   guide_panel = Panel.fit(
       """🎯 Synthcity Detection Results:
• Using proven synthcity detection implementations
• 0.5 = Random guessing (IDEAL - classifier can't tell real from synthetic)
• 0.6 = Slight detection ability (GOOD)  
• 0.7 = Moderate detection ability (FAIR)
• 0.8+ = Strong detection ability (POOR - synthetic data easily detected)

Lower scores = Higher synthetic data quality""",
       title="📖 Synthcity Detection Guide",
       border_style="blue"
   )

   console.print(guide_panel)


if __name__ == "__main__":
    main()
