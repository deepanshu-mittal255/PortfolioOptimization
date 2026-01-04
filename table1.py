import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

# Data
data = {
    "Metric": ["Return Range", "Max Daily Return", "Min Daily Return",
               "Max Log Cumulative Return", "Expected Return (Annualized)",
               "Volatility (Annualized)", "Sharpe Ratio", "Sortino Ratio",
               "Maximum Drawdown", "% Positive Days", "Win/Loss Ratio"],
    "Value": ["-5.71% to +9.34%", "+9.34%", "-5.71%", "+2.37",
              "16.41%", "9.88%", "1.66", "2.23",
              "-14.90%", "55.23%", "1.15"]
}

# Create table
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

table = ax.table(
    cellText=pd.DataFrame(data).values,
    colLabels=["Performance Metric", "Value"],
    loc='center',
    cellLoc='center',
    colColours=['#f0f0f0', '#f0f0f0']
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

# Style header
for (row, col), cell in table.get_celld().items():
    if row == 0:  # Header
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        cell.set_facecolor('#40466e')
        cell.set_text_props(color='white')

plt.title("Deep Learning Strategy Performance Summary (2010-2024)",
          pad=20, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dl_strategy_performance.png', dpi=300, bbox_inches='tight')
plt.show()
