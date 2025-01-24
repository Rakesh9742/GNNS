import pandas as pd
import matplotlib.pyplot as plt

# Path to the predictions log file
log_file_path = r"D:\GNNs\output\predictions_log.txt"

# Load the predictions log file
df = pd.read_csv(log_file_path)

# Preview the data
print("Loaded Data:")
print(df.head())

# Count the number of predictions per label
label_counts = df["Prediction"].value_counts()

# Plot a bar chart for label distribution
plt.figure(figsize=(8, 6))
plt.bar(label_counts.index, label_counts.values, alpha=0.7)
plt.title("Prediction Distribution")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Visualize raw outputs as scatter plot
df["Score_Difference"] = df["Raw_Output"].apply(lambda x: eval(x)[1] - eval(x)[0])

plt.figure(figsize=(10, 6))
plt.scatter(df["ID"], df["Score_Difference"], c=(df["Prediction"] == "Anomaly").map({True: "red", False: "blue"}), alpha=0.5)
plt.title("Raw Model Scores vs ID")
plt.xlabel("ID")
plt.ylabel("Score Difference (Anomaly - Normal)")
plt.axhline(0, color="black", linestyle="--", alpha=0.7)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
