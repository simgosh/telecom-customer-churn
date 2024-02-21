# Import libraries and methods/functions
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


#import data seperating first, then merge them into a df called churn_df 

telecom_demo = pd.read_csv("telecom_demographics.csv")
telecom_usage = pd.read_csv("telecom_usage.csv")

# print(telecom_demo.info())
# print(telecom_usage.info())

churn_df = telecom_demo.merge(telecom_usage, on="customer_id")
print(churn_df.head())

print(churn_df.isnull().sum().sort_values())

churn_df["telecom_partner"] = churn_df["telecom_partner"].astype("category")
churn_df["gender"] = churn_df["gender"].astype("category")
print(churn_df.info())

churn_df["churn"] = churn_df["churn"].astype("bool")
print(churn_df["churn"].dtypes)

print(churn_df[["telecom_partner", "city"]].nunique())

telecom_partner = churn_df.groupby("telecom_partner").agg(avg_sent_sms = ("sms_sent", "mean"),
                                                          avg_made_call =("calls_made", "mean")).sort_values("avg_sent_sms", ascending=False).round(2)

telecom_partner.rename(columns={"index":"Telecom Partner","avg_sent_sms":"Avg Sent SMS", "avg_made_call":"Avg Calls Made"}, inplace=True)
print(telecom_partner)

#TOP 10 Max sent SMS city 

max_sent_sms = churn_df[["sms_sent", "city", "telecom_partner", "gender"]].sort_values("sms_sent", ascending=False).drop_duplicates().reset_index(drop=True)[:11]
print(max_sent_sms)

#Which operator do the people with the highest salaries use? 

max_estimated_salary = churn_df[["telecom_partner", "estimated_salary", "gender", "city", "age"]].sort_values("estimated_salary", ascending=False).drop_duplicates().reset_index(drop=True)[0:20]
max_estimated_salary.rename(columns={"telecom_partner":"Telecom", "estimated_salary":"Salary", "gender":"Sex", "city":"City", "age":"Age"}, inplace=True)
print(max_estimated_salary)
fig = px.scatter(max_estimated_salary, x="Salary", y="Telecom", width=600, height=500)
fig.show()

#Which operator do the people with the lowest salaries use? 
min_estimated_salary = churn_df[["telecom_partner", "estimated_salary", "gender", "city", "age"]].sort_values("estimated_salary", ascending=True).drop_duplicates().reset_index(drop=True)[0:20]
min_estimated_salary.rename(columns={"telecom_partner":"Telecom", "estimated_salary":"Salary", "gender":"Sex", "city":"City", "age":"Age"}, inplace=True)
print(min_estimated_salary)


#Operator review with average & max age 
avg_age_operator = churn_df.groupby("telecom_partner")["age"].agg({"mean","max"})
print(avg_age_operator)

#Convert to registration_event datetime

churn_df["registration_event"] = pd.to_datetime(churn_df["registration_event"])
print(churn_df.dtypes)


#lowest estimated salary groupby telecom-partner 
min_salary=churn_df.groupby("telecom_partner").agg(min_salary = ("estimated_salary", "min")).round(2)
print(min_salary.sort_values("min_salary", ascending=True))


#highest estimated salary groupby telecom-partner 
max_salary=churn_df.groupby("telecom_partner").agg(max_salary = ("estimated_salary", "max")).round(2)
print(max_salary.sort_values("max_salary", ascending=True))

#which city the most sent sms?
city_high = churn_df[["city", "telecom_partner", "sms_sent","gender", "age"]].sort_values("sms_sent", ascending=False).drop_duplicates().reset_index(drop=True)[0:20]
print(city_high)

#which city the most call phone?
city_high = churn_df[["city", "telecom_partner", "calls_made","gender", "age"]].sort_values("calls_made", ascending=False).drop_duplicates().reset_index(drop=True)[0:20]
print(city_high)



#  One Hot Encoding for categorical variables
churn_df = pd.get_dummies(churn_df, columns=["telecom_partner", "gender", "state", "city", "registration_event"])


# Logistic Regression
# Decision Tree
# Ensemble algorithms: Random Forest

#LogisticRegression 
X = churn_df.drop("churn", axis=1)
y = churn_df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print(accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test,y_pred_logreg))

#DescisionTree Model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print(accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


#RandomForest Model 
rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


#KNeighbors Classifier 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
confusion_matr = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matr, annot=True, fmt='d', cbar=True, cmap="Blues")
plt.title("K-Nearest Neighbors Conf Matrix ")
plt.show()
