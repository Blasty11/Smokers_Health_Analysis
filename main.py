import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def plot_scatter(df, x, y, xlabel, ylabel, title):
    plt.scatter(df[x], df[y], alpha=0.5, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_histogram_kde(df, column, bins=20):
    plt.hist(df[column], color='purple', bins=bins, density=True, alpha=0.6)
    sns.kdeplot(df[column], color='blue', linestyle='-', linewidth=2)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f'Distribution of {column}')
    plt.legend()
    plt.show()

def plot_age_groups(df):
    age_groups = pd.cut(df['age'], bins=range(20, 81, 10))
    age_groups.value_counts().sort_index().plot(kind='bar', color='green')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.title('Distribution of Age Groups')
    plt.xticks(rotation=45)
    plt.show()

def plot_boxplot(df, x, y, xlabel, ylabel, title):
    sns.boxplot(x=df[x], y=df[y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_heatmap_correlation(df):
    plt.figure(figsize=(10, 10))
    numeric_df = df._get_numeric_data()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu', fmt=".2f")
    plt.title('Pairwise correlation of columns')
    plt.show()

def plot_pie_chart_age_groups(df):
    plt.figure(figsize=(8, 8))
    age_groups = pd.cut(df['age'], bins=range(20, 81, 10))
    age_groups_counts = age_groups.value_counts().sort_index()
    plt.pie(age_groups_counts, labels=age_groups_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.title('Distribution of Age Groups')
    plt.axis('equal')  
    plt.show()
def perform_linear_regression(df):
    # Fill NaN values as zero
    df.fillna(0, inplace=True)
    
    X = df[['age', 'heart_rate', 'cigs_per_day']]
    y = df['chol']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    model_score = model.score(X, y)
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)
    print("Model Score:", model_score)


def main_menu(df):
    while True:
        print("\nMain Menu:")
        print("1. Plot Scatter (Cigarettes Per Day vs. Cholesterol)")
        print("2. Plot Histogram and KDE (Cholesterol)")
        print("3. Plot Age Groups Distribution")
        print("4. Plot Boxplot (Gender vs. Heart Rate)")
        print("5. Plot Heatmap of Correlation")
        print("6. Plot Pie Chart of Age Groups")
        print("7. Perform Linear Regression Prediction for Current Smokers")
        print("8. Exit")

        choice = input("Enter your choice (1-8): ")

        if choice == '1':
            plot_scatter(df, 'cigs_per_day', 'chol', 'Cigarettes Per Day', 'Cholesterol', 'Cigarettes Per Day vs. Cholesterol')
        elif choice == '2':
            plot_histogram_kde(df, 'chol')
        elif choice == '3':
            plot_age_groups(df)
        elif choice == '4':
            plot_boxplot(df, 'sex', 'heart_rate', 'Gender', 'Heart Rate', 'Heart Rate Distribution by Gender')
        elif choice == '5':
            plot_heatmap_correlation(df)
        elif choice == '6':
            plot_pie_chart_age_groups(df)
        elif choice == '7':
            perform_linear_regression(df)
        elif choice == '8':
            print("Exiting the program...")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 8.")

# Read data
url = "https://github.com/umairaltaf982/Smokers-_Health_Analysis/raw/main/smoking_health_data_final.csv"
df = pd.read_csv(url)

# Main menu
main_menu(df)
