"""
Generate a synthetic student performance dataset.
Mimics realistic relationships: more study hours, higher attendance,
better previous scores and assignment scores -> higher final marks,
with some random noise and a few missing values to simulate real data.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

study_hours = np.clip(np.random.normal(4.5, 2.0, N), 0, 12)
attendance = np.clip(np.random.normal(78, 12, N), 30, 100)
previous_score = np.clip(np.random.normal(65, 15, N), 0, 100)
assignments_score = np.clip(np.random.normal(70, 14, N), 0, 100)

gender = np.random.choice(["Male", "Female"], size=N)
parent_education = np.random.choice(
    ["High School", "Bachelors", "Masters", "PhD"],
    size=N, p=[0.35, 0.35, 0.22, 0.08]
)

# Underlying relationship for Final Marks (with noise)
final_marks = (
    2.8 * study_hours +
    0.28 * attendance +
    0.30 * previous_score +
    0.25 * assignments_score +
    np.random.normal(0, 6, N)
)
final_marks = np.clip(final_marks, 0, 100)

df = pd.DataFrame({
    "Study_Hours": study_hours.round(2),
    "Attendance": attendance.round(1),
    "Previous_Score": previous_score.round(1),
    "Assignments_Score": assignments_score.round(1),
    "Gender": gender,
    "Parent_Education": parent_education,
    "Final_Marks": final_marks.round(1),
})

# Inject some missing values to simulate real-world messiness
for col, frac in [("Attendance", 0.03), ("Assignments_Score", 0.02), ("Previous_Score", 0.02)]:
    idx = df.sample(frac=frac, random_state=1).index
    df.loc[idx, col] = np.nan

df.to_csv("/home/claude/student_perf/data/student_performance.csv", index=False)
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
