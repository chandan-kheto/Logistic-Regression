
# 1.import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2.load the data
data = {
  'Hours_studied': [2, 4, 5, 6, 7, 8, 9, 10],
  'Pass_Exam': [0, 0, 0, 1, 1, 1, 1, 1,]
}
df = pd.DataFrame(data)
x = df[['Hours_studied']] # features
y = df['Pass_Exam']       # target

# 3.split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4.train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# 5.make predictions
y_pred = model.predict(x_test)

# 6.evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("y_test values:", list(y_test))
print("y_pred values:", list(y_pred))
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{report}")

# 7.visualize the data
plt.scatter(x, y)
plt.xlabel('Hours')
plt.ylabel('Pass')
plt.show()

# 8.visualize the model
plt.scatter(x, y)
plt.plot(x, model.predict(x), color='red')
plt.xlabel('Hours')
plt.ylabel('Pass')
plt.show()

# Step 9: Predict new value
new_data = [[10]]
pred = model.predict(new_data)
print('Will a student who studies 10 hours pass?','Yes' if pred[0] == 1 else 'No')

