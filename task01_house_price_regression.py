"""
SkillCraft Technology – Task 01
House Price Prediction using Linear Regression
Features: sqft_living, bedrooms, bathrooms → price
"""

# ─────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# 2. Load & Explore Dataset
# ─────────────────────────────────────────
df = pd.read_csv("house_data.csv")

print("=" * 55)
print("  HOUSE PRICE PREDICTION – LINEAR REGRESSION")
print("=" * 55)
print("\n📋 Dataset Shape:", df.shape)
print("\n📊 First 5 Rows:")
print(df.head().to_string())
print("\n📈 Summary Statistics:")
print(df.describe().round(2).to_string())
print("\n🔍 Missing Values:")
print(df.isnull().sum().to_string())

# ─────────────────────────────────────────
# 3. Feature / Target Split + Train-Test Split
# ─────────────────────────────────────────
FEATURES = ["sqft_living", "bedrooms", "bathrooms"]
TARGET   = "price"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📦 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 4. Feature Scaling
# ─────────────────────────────────────────
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 5. Train Model
# ─────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_s, y_train)

print("\n🤖 Model Coefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"   {feat:>15}: {coef:>12.2f}")
print(f"   {'Intercept':>15}: {model.intercept_:>12.2f}")

# ─────────────────────────────────────────
# 6. Evaluate Model
# ─────────────────────────────────────────
y_pred = model.predict(X_test_s)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n📉 Model Evaluation (Test Set):")
print(f"   MSE  : {mse:>15,.2f}")
print(f"   RMSE : {rmse:>15,.2f}")
print(f"   MAE  : {mae:>15,.2f}")
print(f"   R²   : {r2:>15.4f}")

# ─────────────────────────────────────────
# 7. Sample Predictions
# ─────────────────────────────────────────
print("\n🏠 Sample Predictions:")
samples = pd.DataFrame({
    "sqft_living": [1000, 2500, 4000],
    "bedrooms":    [2,    3,    5],
    "bathrooms":   [1,    2,    4],
})
samples_s = scaler.transform(samples)
preds = model.predict(samples_s)
for i, row in samples.iterrows():
    print(f"   {int(row.sqft_living)} sqft | {int(row.bedrooms)} bed | "
          f"{int(row.bathrooms)} bath → ${preds[i]:,.0f}")

# ─────────────────────────────────────────
# 8. Visualizations
# ─────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
fig = plt.figure(figsize=(16, 12), facecolor="#1a1a2e")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

DARK_BG   = "#1a1a2e"
CARD_BG   = "#16213e"
ACCENT    = "#e94560"
ACCENT2   = "#0f3460"
TEXT_COL  = "#eaeaea"
GRID_COL  = "#2a2a4a"

ax_params = dict(facecolor=CARD_BG)

# ── 8a. Correlation Heatmap ───────────────
ax0 = fig.add_subplot(gs[0, 0], **ax_params)
corr = df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax0,
            annot_kws={"size": 10, "color": TEXT_COL})
ax0.set_title("Correlation Matrix", color=TEXT_COL, fontsize=12, fontweight="bold")
ax0.tick_params(colors=TEXT_COL)
for spine in ax0.spines.values():
    spine.set_edgecolor(GRID_COL)

# ── 8b. Actual vs Predicted ───────────────
ax1 = fig.add_subplot(gs[0, 1], **ax_params)
ax1.scatter(y_test, y_pred, alpha=0.5, color=ACCENT, edgecolors="none", s=25)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax1.plot(lims, lims, "w--", linewidth=1.5, label="Perfect Fit")
ax1.set_xlabel("Actual Price ($)", color=TEXT_COL)
ax1.set_ylabel("Predicted Price ($)", color=TEXT_COL)
ax1.set_title("Actual vs Predicted", color=TEXT_COL, fontsize=12, fontweight="bold")
ax1.tick_params(colors=TEXT_COL)
ax1.legend(labelcolor=TEXT_COL, facecolor=CARD_BG)
for spine in ax1.spines.values():
    spine.set_edgecolor(GRID_COL)

# ── 8c. Residuals Distribution ────────────
ax2 = fig.add_subplot(gs[0, 2], **ax_params)
residuals = y_test - y_pred
ax2.hist(residuals, bins=35, color=ACCENT2, edgecolor=ACCENT, linewidth=0.5, alpha=0.85)
ax2.axvline(0, color="white", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Residual ($)", color=TEXT_COL)
ax2.set_ylabel("Frequency", color=TEXT_COL)
ax2.set_title("Residuals Distribution", color=TEXT_COL, fontsize=12, fontweight="bold")
ax2.tick_params(colors=TEXT_COL)
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID_COL)

# ── 8d. sqft_living vs Price ──────────────
ax3 = fig.add_subplot(gs[1, 0], **ax_params)
ax3.scatter(df["sqft_living"], df["price"], alpha=0.4,
            color="#53d8fb", edgecolors="none", s=20)
# regression line
sqft_range = np.linspace(df["sqft_living"].min(), df["sqft_living"].max(), 100)
# fix bedrooms & bathrooms at their means
mean_bed  = df["bedrooms"].mean()
mean_bath = df["bathrooms"].mean()
line_data = pd.DataFrame({
    "sqft_living": sqft_range,
    "bedrooms":    np.full(100, mean_bed),
    "bathrooms":   np.full(100, mean_bath),
})
line_preds = model.predict(scaler.transform(line_data))
ax3.plot(sqft_range, line_preds, color=ACCENT, linewidth=2, label="Regression Line")
ax3.set_xlabel("Sqft Living", color=TEXT_COL)
ax3.set_ylabel("Price ($)", color=TEXT_COL)
ax3.set_title("Sqft Living vs Price", color=TEXT_COL, fontsize=12, fontweight="bold")
ax3.tick_params(colors=TEXT_COL)
ax3.legend(labelcolor=TEXT_COL, facecolor=CARD_BG)
for spine in ax3.spines.values():
    spine.set_edgecolor(GRID_COL)

# ── 8e. Feature Importance (Coefficients) ─
ax4 = fig.add_subplot(gs[1, 1], **ax_params)
feat_coefs = pd.Series(model.coef_, index=FEATURES).sort_values()
colors_bar = [ACCENT if c < 0 else "#53d8fb" for c in feat_coefs]
feat_coefs.plot(kind="barh", ax=ax4, color=colors_bar, edgecolor="none")
ax4.set_xlabel("Coefficient Value", color=TEXT_COL)
ax4.set_title("Feature Coefficients\n(Standardised)", color=TEXT_COL,
              fontsize=12, fontweight="bold")
ax4.tick_params(colors=TEXT_COL)
ax4.axvline(0, color="white", linewidth=0.8, linestyle="--")
for spine in ax4.spines.values():
    spine.set_edgecolor(GRID_COL)

# ── 8f. Metrics Card ──────────────────────
ax5 = fig.add_subplot(gs[1, 2], **ax_params)
ax5.axis("off")
metrics_text = (
    f"  Model Performance\n"
    f"  {'─'*26}\n\n"
    f"  R² Score     :  {r2:.4f}\n\n"
    f"  RMSE         :  ${rmse:>10,.0f}\n\n"
    f"  MAE          :  ${mae:>10,.0f}\n\n"
    f"  MSE          :  ${mse:>10,.0f}\n\n"
    f"  Train Rows   :  {len(X_train)}\n\n"
    f"  Test Rows    :  {len(X_test)}"
)
ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes,
         verticalalignment="top", fontsize=11,
         fontfamily="monospace", color=TEXT_COL,
         bbox=dict(boxstyle="round,pad=0.6", facecolor=ACCENT2,
                   edgecolor=ACCENT, linewidth=1.5))
ax5.set_title("Summary", color=TEXT_COL, fontsize=12, fontweight="bold")

# ── Super title ───────────────────────────
fig.suptitle("🏠  House Price Prediction — Linear Regression  |  SkillCraft Task 01",
             color=TEXT_COL, fontsize=14, fontweight="bold", y=0.98)
fig.patch.set_facecolor(DARK_BG)

plt.savefig("/mnt/user-data/outputs/task01_results.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print("\n✅ Plot saved to task01_results.png")
