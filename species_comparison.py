import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(manual_file, ai_file):
    """Load and clean the Excel files"""
    # Load the files (adjust column names as needed)
    manual_df = pd.read_excel(manual_file)
    ai_df = pd.read_excel(ai_file)
    
    # Get the first column (assuming species names are in the first column)
    manual_species = manual_df.iloc[:, 0].dropna().astype(str).str.strip()
    ai_species = ai_df.iloc[:, 0].dropna().astype(str).str.strip()
    
    # Convert to sets for comparison (removing duplicates and empty strings)
    manual_set = set(manual_species[manual_species != ''])
    ai_set = set(ai_species[ai_species != ''])
    
    return manual_species, ai_species, manual_set, ai_set

def calculate_metrics(manual_set, ai_set):
    """Calculate comparison metrics"""
    # Basic set operations
    intersection = manual_set.intersection(ai_set)
    manual_only = manual_set - ai_set
    ai_only = ai_set - manual_set
    union = manual_set.union(ai_set)
    
    # Calculate metrics
    metrics = {
        'Total Manual Species': len(manual_set),
        'Total AI Species': len(ai_set),
        'Common Species': len(intersection),
        'Manual Only': len(manual_only),
        'AI Only': len(ai_only),
        'Total Unique Species': len(union)
    }
    
    # Calculate efficiency metrics (assuming manual extraction is ground truth)
    if len(manual_set) > 0:
        recall = len(intersection) / len(manual_set)  # How many manual species were found by AI
        metrics['Recall (Sensitivity)'] = recall
    else:
        metrics['Recall (Sensitivity)'] = 0
    
    if len(ai_set) > 0:
        precision = len(intersection) / len(ai_set)  # How many AI species were correct
        metrics['Precision'] = precision
    else:
        metrics['Precision'] = 0
    
    # F1 Score
    if metrics['Precision'] + metrics['Recall (Sensitivity)'] > 0:
        f1_score = 2 * (metrics['Precision'] * metrics['Recall (Sensitivity)']) / (metrics['Precision'] + metrics['Recall (Sensitivity)'])
        metrics['F1-Score'] = f1_score
    else:
        metrics['F1-Score'] = 0
    
    # Jaccard Index (Intersection over Union)
    if len(union) > 0:
        jaccard = len(intersection) / len(union)
        metrics['Jaccard Index'] = jaccard
    else:
        metrics['Jaccard Index'] = 0
    
    return metrics, intersection, manual_only, ai_only

def create_visualizations(metrics, manual_set, ai_set, intersection, manual_only, ai_only):
    """Create comprehensive visualizations"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Venn Diagram-like Bar Chart
    ax1 = plt.subplot(2, 4, 1)
    categories = ['Manual Only', 'Common', 'AI Only']
    values = [len(manual_only), len(intersection), len(ai_only)]
    colors = ['#ff7f7f', '#7fbf7f', '#7f7fff']
    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Species Distribution Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Species')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie Chart of Total Distribution
    ax2 = plt.subplot(2, 4, 2)
    labels = ['Manual Only', 'Common', 'AI Only']
    sizes = [len(manual_only), len(intersection), len(ai_only)]
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Species Distribution (%)', fontsize=12, fontweight='bold')
    
    # 3. Performance Metrics Bar Chart
    ax3 = plt.subplot(2, 4, 3)
    perf_metrics = ['Precision', 'Recall (Sensitivity)', 'F1-Score', 'Jaccard Index']
    perf_values = [metrics[m] for m in perf_metrics]
    bars = ax3.bar(perf_metrics, perf_values, color='skyblue', alpha=0.7, edgecolor='navy')
    ax3.set_title('AI Performance Metrics', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, perf_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Total Species Count Comparison
    ax4 = plt.subplot(2, 4, 4)
    methods = ['Manual\nExtraction', 'AI\nExtraction']
    totals = [len(manual_set), len(ai_set)]
    bars = ax4.bar(methods, totals, color=['orange', 'green'], alpha=0.7, edgecolor='black')
    ax4.set_title('Total Species Count', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Species')
    
    # Add value labels on bars
    for bar, value in zip(bars, totals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # 5. Overlap Analysis
    ax5 = plt.subplot(2, 4, 5)
    overlap_data = {
        'Total Manual': len(manual_set),
        'Total AI': len(ai_set),
        'Overlap': len(intersection),
        'Manual Missed by AI': len(manual_only),
        'AI False Positives': len(ai_only)
    }
    x_pos = range(len(overlap_data))
    bars = ax5.bar(x_pos, list(overlap_data.values()), 
                   color=['coral', 'lightgreen', 'gold', 'lightcoral', 'lightblue'],
                   alpha=0.7, edgecolor='black')
    ax5.set_title('Detailed Overlap Analysis', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(list(overlap_data.keys()), rotation=45, ha='right')
    ax5.set_ylabel('Count')
    
    # Add value labels on bars
    for bar, value in zip(bars, overlap_data.values()):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # 6. Efficiency Radar Chart
    ax6 = plt.subplot(2, 4, 6, projection='polar')
    angles = np.linspace(0, 2*np.pi, len(perf_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    values = perf_values + perf_values[:1]  # Complete the circle
    
    ax6.plot(angles, values, 'o-', linewidth=2, color='red', alpha=0.7)
    ax6.fill(angles, values, alpha=0.25, color='red')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(perf_metrics)
    ax6.set_ylim(0, 1)
    ax6.set_title('AI Performance Radar', fontsize=12, fontweight='bold', pad=20)
    
    # 7. Comparative Analysis
    ax7 = plt.subplot(2, 4, 7)
    comparison_metrics = ['Coverage\n(AI/Manual)', 'Precision', 'Extra Species\n(AI only)']
    coverage = len(ai_set) / len(manual_set) if len(manual_set) > 0 else 0
    extra_ratio = len(ai_only) / len(manual_set) if len(manual_set) > 0 else 0
    comp_values = [coverage, metrics['Precision'], extra_ratio]
    
    bars = ax7.bar(comparison_metrics, comp_values, 
                   color=['purple', 'teal', 'orange'], alpha=0.7, edgecolor='black')
    ax7.set_title('Comparative Metrics', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Ratio/Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, comp_values):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Summary Statistics Table
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('tight')
    ax8.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', 'Value'],
        ['Total Manual Species', f"{metrics['Total Manual Species']}"],
        ['Total AI Species', f"{metrics['Total AI Species']}"],
        ['Common Species', f"{metrics['Common Species']}"],
        ['Precision', f"{metrics['Precision']:.3f}"],
        ['Recall', f"{metrics['Recall (Sensitivity)']:.3f}"],
        ['F1-Score', f"{metrics['F1-Score']:.3f}"],
        ['Jaccard Index', f"{metrics['Jaccard Index']:.3f}"]
    ]
    
    table = ax8.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax8.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('species_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(metrics, intersection, manual_only, ai_only, manual_set, ai_set):
    """Generate a detailed text report"""
    
    print("=" * 80)
    print("SPECIES EXTRACTION COMPARISON REPORT")
    print("=" * 80)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   • Manual extraction found: {len(manual_set)} unique species")
    print(f"   • AI extraction found: {len(ai_set)} unique species")
    print(f"   • Species found by both methods: {len(intersection)}")
    print(f"   • Species found only manually: {len(manual_only)}")
    print(f"   • Species found only by AI: {len(ai_only)}")
    print(f"   • Total unique species across both methods: {len(manual_set.union(ai_set))}")
    
    print(f"\n🎯 PERFORMANCE METRICS (AI vs Manual as Ground Truth):")
    print(f"   • Precision: {metrics['Precision']:.3f} ({metrics['Precision']*100:.1f}%)")
    print(f"     - Of all species AI found, {metrics['Precision']*100:.1f}% were correct")
    print(f"   • Recall (Sensitivity): {metrics['Recall (Sensitivity)']:.3f} ({metrics['Recall (Sensitivity)']*100:.1f}%)")
    print(f"     - AI found {metrics['Recall (Sensitivity)']*100:.1f}% of manually identified species")
    print(f"   • F1-Score: {metrics['F1-Score']:.3f}")
    print(f"     - Harmonic mean of precision and recall")
    print(f"   • Jaccard Index: {metrics['Jaccard Index']:.3f}")
    print(f"     - Similarity coefficient (intersection/union)")
    
    print(f"\n📈 EFFICIENCY ANALYSIS:")
    coverage_ratio = len(ai_set) / len(manual_set) if len(manual_set) > 0 else 0
    print(f"   • Coverage Ratio: {coverage_ratio:.3f}")
    print(f"     - AI found {coverage_ratio*100:.1f}% as many species as manual method")
    
    if len(ai_only) > 0:
        print(f"   • AI discovered {len(ai_only)} additional species not found manually")
        print(f"   • This could indicate either:")
        print(f"     - AI found species that manual extraction missed (good)")
        print(f"     - AI produced false positives (needs verification)")
    
    print(f"\n🔍 DETAILED BREAKDOWN:")
    print(f"   • Agreement rate: {len(intersection)/len(manual_set.union(ai_set))*100:.1f}%")
    print(f"   • Manual method missed: {len(ai_only)} species that AI found")
    print(f"   • AI method missed: {len(manual_only)} species that manual found")
    
    # Performance interpretation
    print(f"\n💡 INTERPRETATION:")
    if metrics['Precision'] >= 0.8:
        print(f"   ✅ HIGH PRECISION: AI rarely produces false positives")
    elif metrics['Precision'] >= 0.6:
        print(f"   ⚠️  MODERATE PRECISION: Some AI results may need verification")
    else:
        print(f"   ❌ LOW PRECISION: Many AI results may be incorrect")
    
    if metrics['Recall (Sensitivity)'] >= 0.8:
        print(f"   ✅ HIGH RECALL: AI catches most species that manual method found")
    elif metrics['Recall (Sensitivity)'] >= 0.6:
        print(f"   ⚠️  MODERATE RECALL: AI misses some species found manually")
    else:
        print(f"   ❌ LOW RECALL: AI misses many species found manually")
    
    if metrics['F1-Score'] >= 0.8:
        print(f"   ✅ EXCELLENT OVERALL PERFORMANCE")
    elif metrics['F1-Score'] >= 0.6:
        print(f"   ⚠️  GOOD OVERALL PERFORMANCE")
    else:
        print(f"   ❌ PERFORMANCE NEEDS IMPROVEMENT")
    
    print("=" * 80)

def export_detailed_results(intersection, manual_only, ai_only, metrics):
    """Export detailed results to Excel file"""
    
    with pd.ExcelWriter('species_comparison_results.xlsx', engine='openpyxl') as writer:
        # Common species
        if intersection:
            pd.DataFrame(list(intersection), columns=['Common Species']).to_excel(
                writer, sheet_name='Common Species', index=False)
        
        # Manual only species
        if manual_only:
            pd.DataFrame(list(manual_only), columns=['Manual Only']).to_excel(
                writer, sheet_name='Manual Only', index=False)
        
        # AI only species
        if ai_only:
            pd.DataFrame(list(ai_only), columns=['AI Only']).to_excel(
                writer, sheet_name='AI Only', index=False)
        
        # Metrics summary
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_df.to_excel(writer, sheet_name='Metrics Summary', index=False)
    
    print(f"\n📁 Detailed results exported to 'species_comparison_results.xlsx'")

def main():
    """Main function to run the complete analysis"""
    
    # File paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    manual_file = 'nimya_extract.xlsx'  # Replace with your manual extraction file
    ai_file = 'auto_extract.xlsx'          # Replace with your AI extraction file
    
    try:
        print("Loading and processing data...")
        manual_species, ai_species, manual_set, ai_set = load_and_clean_data(manual_file, ai_file)
        
        print("Calculating metrics...")
        metrics, intersection, manual_only, ai_only = calculate_metrics(manual_set, ai_set)
        
        print("Generating visualizations...")
        create_visualizations(metrics, manual_set, ai_set, intersection, manual_only, ai_only)
        
        print("Generating detailed report...")
        generate_detailed_report(metrics, intersection, manual_only, ai_only, manual_set, ai_set)
        
        print("Exporting detailed results...")
        export_detailed_results(intersection, manual_only, ai_only, metrics)
        
        print("\n✅ Analysis complete! Check the generated files:")
        print("   • species_comparison_analysis.png (visualizations)")
        print("   • species_comparison_results.xlsx (detailed results)")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find the specified file. Please check the file paths.")
        print(f"   Make sure you have files named: '{manual_file}' and '{ai_file}'")
        print(f"   Or update the file paths in the main() function.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()