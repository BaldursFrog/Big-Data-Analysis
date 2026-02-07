import pandas as pd
import json
import os

df = pd.read_csv('processed_amazon_reviews.csv')
subset_df = df.sample(n=1000, random_state=42)
tasks = []
for idx, row in subset_df.iterrows():
    task = {
        "data": {
            "text": row['reviewText'],
            "rating": row['overall']
        },
        "annotations": []  
    }
    tasks.append(task)

with open('label_studio_tasks.json', 'w', encoding='utf-8') as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print("Label Studio tasks saved to 'label_studio_tasks.json'")
print("\nInstructions for Label Studio setup:")
print("1. Install Label Studio: pip install label-studio")
print("2. Run: label-studio start")
print("3. Create a new project with task type 'Text Classification'")
print("4. Import tasks from 'label_studio_tasks.json'")
print("5. Configure labels: 'positive' and 'negative'")
print("6. Manually label some samples, then use ML backend for auto-labeling")
print("7. Export labeled data as JSON")