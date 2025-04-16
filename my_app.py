import streamlit as st
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def preprocess_data(df):
    features = df.drop(columns=["HAS_NEXT_ENROLLMENT", "STUDENTID", "ENTRYDATE", "EXITDATE"])
    target = df["HAS_NEXT_ENROLLMENT"]

    categorical_cols = features.select_dtypes(include=["object"]).columns
    label_encoders = {col: LabelEncoder() for col in categorical_cols}

    for col in categorical_cols:
        features[col] = label_encoders[col].fit_transform(features[col].astype(str))

    imputer = SimpleImputer(strategy="most_frequent")
    features_imputed = imputer.fit_transform(features)

    return features_imputed, target

def main():
    st.title('Urbandale Community School District Re-enrollment Prediction')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['ENTRYDATE', 'EXITDATE'])
            df['STUDENTID'] = df['STUDENTID'].astype(str)

            st.write("Data Preview (First 50 Rows):")
            st.write(df.head(50))

            features_imputed, target = preprocess_data(df)

            X_train, X_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results_df = pd.DataFrame({'STUDENTID': df.iloc[y_test.index]['STUDENTID'], 'Re-enrollment Prediction': y_pred})
            st.write("Re-enrollment Predictions:")
            st.write(results_df)

            total_students = len(y_test)
            total_reenrollments = sum(y_pred)
            reenrollment_percentage = (total_reenrollments / total_students) * 100

            st.write(f"Out of {total_students} students, {total_reenrollments} are predicted to re-enroll.")
            st.write(f"Percentage of students predicted to re-enroll: {reenrollment_percentage:.2f}%")

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

            results_df['Accuracy'] = accuracy
            results_df['Precision'] = precision
            results_df['Recall'] = recall
            results_df['F1 Score'] = f1

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

            st.write("=== Model Performance ===")
            st.write(f"Accuracy:  {accuracy:.4f}")
            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall:    {recall:.4f}")
            st.write(f"F1 Score:  {f1:.4f}")
            st.write("\nClassification Report:\n")
            st.write(classification_report(y_test, y_pred))
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            st.write(f"Exception details: {e}")

if __name__ == "__main__":
    try:
        main()
        st.write("Streamlit app is running...")
    except Exception as e:
        st.error(f"Failed to start Streamlit app: {e}")
        st.write(f"Exception details: {e}")
    finally:
        st.write("Execution completed.")
        input("Press Enter to close the application...")
