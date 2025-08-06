# inspect_model.py
import joblib

def main():
    clf = joblib.load("baseline_logreg.joblib")
    print("Model:", clf)
    print("Coefficients:", clf.coef_)
    print("Intercept:", clf.intercept_)

if __name__ == "__main__":
    main()
