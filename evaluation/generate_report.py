"""
Generate comparison report and visualizations
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """Load RAGAS results"""
    with open("./evaluation/ragas_results.json") as f:
        results = json.load(f)
    return results

def create_comparison_table(results):
    """Create comparison table"""
    df = pd.DataFrame(results).T
    df['average'] = df.mean(axis=1)
    
    # Round to 3 decimal places
    df = df.round(3)
    
    return df

def create_bar_chart(df):
    """Create bar chart visualization"""
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot
    df_plot = df.drop('average', axis=1)
    df_plot.plot(kind='bar', ax=ax, width=0.8)
    
    # Customize
    ax.set_title("RAGAS Metrics Comparison", fontsize=16, fontweight='bold')
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig("./evaluation/comparison_chart.png", dpi=300, bbox_inches='tight')
    print("✅ Chart saved to ./evaluation/comparison_chart.png")
    
    plt.close()

def create_heatmap(df):
    """Create heatmap visualization"""
    # Set style
    sns.set_style("white")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot heatmap
    df_heatmap = df.drop('average', axis=1)
    sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1)
    
    # Customize
    ax.set_title("RAGAS Metrics Heatmap", fontsize=16, fontweight='bold')
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig("./evaluation/heatmap.png", dpi=300, bbox_inches='tight')
    print("✅ Heatmap saved to ./evaluation/heatmap.png")
    
    plt.close()

def generate_report(df):
    """Generate text report"""
    report = []
    report.append("=" * 60)
    report.append("AI HEALTHCARE AGENT - EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall comparison
    report.append("OVERALL COMPARISON")
    report.append("-" * 60)
    report.append(df.to_string())
    report.append("")
    
    # Winner
    winner = df['average'].idxmax()
    winner_score = df.loc[winner, 'average']
    report.append(f"🏆 WINNER: {winner.upper()}")
    report.append(f"   Average Score: {winner_score:.3f}")
    report.append("")
    
    # Detailed analysis
    report.append("DETAILED ANALYSIS")
    report.append("-" * 60)
    
    for model in df.index:
        report.append(f"\n{model.upper()}:")
        report.append(f"  Faithfulness:       {df.loc[model, 'faithfulness']:.3f}")
        report.append(f"  Answer Relevancy:   {df.loc[model, 'answer_relevancy']:.3f}")
        report.append(f"  Context Precision:  {df.loc[model, 'context_precision']:.3f}")
        report.append(f"  Context Recall:     {df.loc[model, 'context_recall']:.3f}")
        report.append(f"  Answer Correctness: {df.loc[model, 'answer_correctness']:.3f}")
        report.append(f"  Average:            {df.loc[model, 'average']:.3f}")
    
    report.append("")
    report.append("=" * 60)
    
    # Save report
    report_text = "\n".join(report)
    with open("./evaluation/report.txt", "w") as f:
        f.write(report_text)
    
    print("✅ Report saved to ./evaluation/report.txt")
    
    # Print to console
    print("\n" + report_text)

def main():
    """Generate comparison report"""
    print("=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)
    
    # Load results
    results = load_results()
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Create visualizations
    create_bar_chart(df)
    create_heatmap(df)
    
    # Generate text report
    generate_report(df)
    
    print("\n✅ All reports generated!")
    print("   - comparison_table.csv")
    print("   - comparison_chart.png")
    print("   - heatmap.png")
    print("   - report.txt")

if __name__ == "__main__":
    main()
