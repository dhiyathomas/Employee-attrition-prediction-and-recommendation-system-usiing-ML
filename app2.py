from flask import Flask, flash, redirect, render_template, request, session
from cs50 import SQL
import sqlite3
from flask_session import Session
from tempfile import mkdtemp
from helpers import login_required, sorry, updateMessage
from machine import run
import pandas as pd 
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import threading
import matplotlib
import matplotlib.pyplot as plt

app = Flask(__name__)


app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-re-validate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route("/")
@login_required
def index():
    return render_template("login.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    db = SQL("sqlite:///hr.db")
    session.clear()
    if request.method == "POST":
        if not request.form.get("username"):
            return sorry("must provide username")
        elif not request.form.get("password"):
            return sorry("must provide password")
        rows = db.execute("SELECT * FROM login WHERE username = :username", username=request.form.get("username"))
        if len(rows) != 1 or rows[0]["password"] != request.form.get("password"):
            return sorry("invalid username and/or password")
        session["user_id"] = rows[0]["id"]
        return redirect("/upload")
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route('/upload')
def upload():
    return render_template('upload.html')  

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'utf-8')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df) 


@app.route("/home")
@login_required
def home():
    return render_template("home.html")


@app.route("/view")
@login_required
def view():
    conn = sqlite3.connect("dataset.db")
    cur = conn.execute("SELECT * FROM dataset")
    data = cur.fetchall()
    return render_template("view.html", data=data)


@app.route("/update")
@login_required
def update():
    return render_template("update.html")


@app.route("/updateDet")
@login_required
def updateDet():
    if not request.values.get("id"):
        return updateMessage("must enter ID")
    id = request.values.get("id")
    param = request.values.get("parameter")
    paramval = request.values.get("paramvalue")
    job = request.values.get("JobRole")
    marstat = request.values.get("MaritalStatus")
    if int(id) < 0 or int(id) > 29:
        return updateMessage("ID not found")
    db = SQL("sqlite:///dataset.db")
    if param == "":
        if job == "":
            db.execute("UPDATE dataset SET MaritalStatus=:ms WHERE ID=:i", ms=marstat, i=int(id))
        elif marstat == "":
            db.execute("UPDATE dataset SET JobRole=:j WHERE ID=:i", j=job, i=int(id))
        else:
            db.execute("UPDATE dataset SET  JobRole=:j, MaritalStatus=:ms WHERE ID=:i", j=job, ms=marstat, i=int(id))
    else:
        if job == "" and marstat == "":
            db.execute("UPDATE dataset SET " + param + "=:val WHERE ID=:i", val=int(paramval), i=int(id))
        elif job == "":
            db.execute("UPDATE dataset SET " + param + "=:val, MaritalStatus=:ms WHERE ID=:i", val=int(paramval), ms=marstat, i=int(id))
        elif marstat == "":
            db.execute("UPDATE dataset SET " + param + "=:val, JobRole=:j WHERE ID=:i", val=int(paramval), j=job, i=id)
        else:
            db.execute("UPDATE dataset SET " + param + "=:val, JobRole=:j, MaritalStatus=:ms WHERE ID=:i", val=int(paramval), j=job, ms=marstat, i=int(id))
    return updateMessage("Updated successfully")


@app.route("/attrition")
@login_required
def attrition():
    conn = sqlite3.connect("dataset.db")
    cur = conn.execute("SELECT * FROM dataset")
    data = cur.fetchall()
    att = run(data)
    retention=recommend()
    print(retention)
    newd = []
    i=0
    for row in range(len(retention)):
        t = ()
        t = t + (i,)
        r=retention.loc[row,'Attrition']
        s=retention.loc[row,'RetentionFactor']
        print(t)
        t = t + (r,)+(s,)
        i = i+1
        #print(t)
        newd.append(t)
    return render_template("attrition.html", data=newd)

def recommend():
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    # Load and preprocess the employee attrition dataset
    data = pd.read_csv('dataset1.csv')  # Replace with the actual path to your dataset
    # Perform data preprocessing, such as handling missing values, encoding categorical variables, etc.

    # Split the dataset into features (X) and target (y)
    X = data.drop(['Attrition','BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'], axis=1)
    y = data['Attrition']  # Target


    # Train a decision tree classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X, y)

    # Make predictions for employee attrition
    predictions = classifier.predict(X)

    # Create a recommendation dataframe to store retention factors for each employee
    recommendation_df = pd.DataFrame()

    # Loop through each employee in the dataset
    for i in range(len(data)):
        employee_id = data['EmployeeID'][i]
        retention_factor = ''
        if predictions[i] == 'Yes':  # If attrition is predicted
            # Based on the decision tree classifier, recommend retention factors
            if data['YearsInCurrentRole'][i] < 2:
                retention_factor = 'Provide mentoring and training opportunities to enhance skills'
            elif data['JobSatisfaction'][i] < 3:
                retention_factor = 'Conduct job satisfaction surveys and address the issues'
            elif data['Age'][i] > 30:
                    retention_factor += 'Provide work-life balance initiatives and flexible working arrangements\n'
            elif data['DistanceFromHome'][i] > 15:
                retention_factor += 'Offer relocation assistance or remote work options\n'
            elif data['JobSatisfaction'][i] < 4:
                retention_factor += 'Conduct job satisfaction surveys and address the issues\n'
            elif data['MonthlyIncome'][i] < 5000:
                retention_factor += 'Provide competitive salaries and benefits\n'
            elif data['NumCompaniesWorked'][i] > 5:
                retention_factor += 'Offer opportunities for job rotation and internal mobility\n'
            elif data['PercentSalaryHike'][i] < 16:
                retention_factor += 'Offer salary hike to employees\n'
            elif data['YearsInCurrentRole'][i] < 2:
                retention_factor += 'Provide mentoring and training opportunities to enhance skills\n'
            elif data['YearsSinceLastPromotion'][i] > 2:
                retention_factor += 'Offer career development opportunities and promotions\n'
            elif data['YearsWithCurrManager'][i] < 2:
                retention_factor += 'Provide opportunities for employee feedback and career growth\n'
            else:
                retention_factor = 'Offer career development opportunities and promotions'
        else:
            retention_factor = 'No retention factors recommended'  # If no attrition is predicted

        recommendation_df = recommendation_df.append({'Attrition':predictions[i], 'RetentionFactor': retention_factor}, ignore_index=True)

    # Print the recommendation dataframe
    #print(recommendation_df)
    return recommendation_df
"""
def recommend():
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    # Load and preprocess the employee attrition dataset
    data = pd.read_csv('dataset1.csv')  # Replace with the actual path to your dataset
    # Perform data preprocessing, such as handling missing values, encoding categorical variables, etc.

    # Split the dataset into features (X) and target (y)
    X = data.drop(['Attrition'], axis=1)
    y = data['Attrition']  # Target

    # Train a decision tree classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X, y)

    # Make predictions for employee attrition
    predictions = classifier.predict(X)

    # Create a recommendation dataframe to store retention factors for each employee
    recommendation_df = pd.DataFrame()

    # Loop through each employee in the dataset
    for i in range(len(data)):
        employee_id = data['EmployeeID'][i]
        retention_factor = ''
        if predictions[i] == 'Yes':  # If attrition is predicted
            # Based on the decision tree classifier, recommend retention factors for each feature in the dataset
            if data['Age'][i] > 30:
                retention_factor += 'Provide work-life balance initiatives and flexible working arrangements\n'
            if data['DistanceFromHome'][i] > 15:
                retention_factor += 'Offer relocation assistance or remote work options\n'
            if data['JobSatisfaction'][i] < 4:
                retention_factor += 'Conduct job satisfaction surveys and address the issues\n'
            if data['MonthlyIncome'][i] < 5000:
                retention_factor += 'Provide competitive salaries and benefits\n'
            if data['NumCompaniesWorked'][i] > 5:
                retention_factor += 'Offer opportunities for job rotation and internal mobility\n'
            if data['PercentSalaryHike'][i] < 16:
                retention_factor += 'Offer salary hike to employees\n'
            if data['YearsInCurrentRole'][i] < 2:
                retention_factor += 'Provide mentoring and training opportunities to enhance skills\n'
            if data['YearsSinceLastPromotion'][i] > 2:
                retention_factor += 'Offer career development opportunities and promotions\n'
            if data['YearsWithCurrManager'][i] < 2:
                retention_factor += 'Provide opportunities for employee feedback and career growth\n'
        else:
            retention_factor = 'No retention factors recommended'  # If no attrition is predicted

        recommendation_df = recommendation_df.append({'Attrition': predictions[i], 'RetentionFactor': retention_factor}, ignore_index=True)

    # Print the recommendation dataframe
    #print(recommendation_df)
    return recommendation_df

"""

@app.route("/graph")
@login_required
def graph():
    return render_template("graph.html")

"""
# Filter the data to show only "Yes" values in the "Attrition" column
    attrition_data = data[data['Attrition'] == 'Yes']

    # Calculate the count of attrition by department
    attrition_by = attrition_data.groupby(['Department']).size().reset_index(name='Count')

    # Create a donut chart
    fig = go.Figure(data=[go.Pie(
        labels=attrition_by['Department'],
        values=attrition_by['Count'],
        hole=0.4,
        marker=dict(colors=['#3CAEA3', '#F6D55C']),
        textposition='inside'
    )])

    # Update the layout
    fig.update_layout(title='Attrition by Department', font=dict(size=16), legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ))

    # Show the chart
    fig.show()
    """
    



if __name__ == '__main__':
         app.run(debug=True)

