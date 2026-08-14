"""
Step 2 & 3: Data Preprocessing + Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("/home/claude/student_perf/data/student_performance.csv")

print("=== BEFORE CLEANING ===")
print(df.isna().sum())

# ---- Handle missing values: fill numeric cols with median ----
num_cols = ["Study_Hours", "Attendance", "Previous_Score", "Assignments_Score"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

print("\n=== AFTER CLEANING ===")
print(df.isna().sum())

# ---- Encode categorical columns ----
df_encoded = df.copy()
df_encoded["Gender"] = df_encoded["Gender"].map({"Male": 0, "Female": 1})

edu_order = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
df_encoded["Parent_Education"] = df_encoded["Parent_Education"].map(edu_order)

df_encoded.to_csv("/home/claude/student_perf/data/student_performance_cleaned.csv", index=False)

# =========== EDA ===========

# 1. Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df_encoded.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Student Features")
plt.tight_layout()
plt.savefig("/home/claude/student_perf/plots/01_correlation_heatmap.png", dpi=120)
plt.close()

# 2. Study Hours vs Final Marks (scatter)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Study_Hours", y="Final_Marks", alpha=0.6)
sns.regplot(data=df, x="Study_Hours", y="Final_Marks", scatter=False, color="red")
plt.title(f"Study Hours vs Final Marks (corr = {df['Study_Hours'].corr(df['Final_Marks']):.2f})")
plt.tight_layout()
plt.savefig("/home/claude/student_perf/plots/02_study_hours_vs_marks.png", dpi=120)
plt.close()

# 3. Attendance vs Final Marks
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Attendance", y="Final_Marks", alpha=0.6, color="green")
sns.regplot(data=df, x="Attendance", y="Final_Marks", scatter=False, color="red")
plt.title(f"Attendance vs Final Marks (corr = {df['Attendance'].corr(df['Final_Marks']):.2f})")
plt.tight_layout()
plt.savefig("/home/claude/student_perf/plots/03_attendance_vs_marks.png", dpi=120)
plt.close()

# 4. Distribution of Final Marks
plt.figure(figsize=(7, 5))
sns.histplot(df["Final_Marks"], bins=25, kde=True, color="purple")
plt.title("Distribution of Final Marks")
plt.xlabel("Final Marks")
plt.tight_layout()
plt.savefig("/home/claude/student_perf/plots/04_final_marks_distribution.png", dpi=120)
plt.close()

# 5. Bar plot: Average marks by Parent Education
plt.figure(figsize=(7, 5))
order = ["High School", "Bachelors", "Masters", "PhD"]
sns.barplot(data=df, x="Parent_Education", y="Final_Marks", order=order, errorbar="sd")
plt.title("Average Final Marks by Parent Education")
plt.tight_layout()
plt.savefig("/home/claude/student_perf/plots/05_marks_by_parent_education.png", dpi=120)
plt.close()

# 6. Boxplot: Marks by Gender
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="Gender", y="Final_Marks")
plt.title("Final Marks Distribution by Gender")
plt.tight_layout()
plt.savefig("/home/claude/student_perf/plots/06_marks_by_gender.png", dpi=120)
plt.close()

print("\nEDA plots saved to /home/claude/student_perf/plots/")
print("\nKey correlations with Final_Marks:")
print(corr["Final_Marks"].sort_values(ascending=False))
