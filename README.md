# Data Analytics Project: Bike Sharing Dataset

This project is an assignment from Dicoding, focusing on data analytics using the Bike Sharing dataset. I conducted the data analysis process in Google Colab, while the dashboard was developed in Visual Studio Code with Streamlit.

The Bike Sharing Dataset comprises hourly and daily counts of rental bikes from the Capital Bikeshare system between the years 2011 and 2012. Although there are two datasets available—hourly and daily—I chose to utilize the hourly dataset for my analysis. This decision allows for a more granular understanding of bike rental patterns influenced by various factors, including weather and time of day.

The dataset provides a comprehensive overview, including data on rental counts, weather conditions, and time attributes. Key attributes include the date, season, year, month, and hour, alongside indicators of holidays and working days. Additionally, weather conditions are categorized, and normalized values for temperature, humidity, and wind speed are provided, along with counts for casual and registered users. This structured information facilitates an in-depth exploration of how different elements affect bike-sharing usage.


## 1. File Structures

```
.
├── dashboard
│   ├── dashboard.py
│   └── hour.csv
│   └── logo.gif
├── data
│   ├── Readme.txt
|   └── hour.csv
├── README.md
├── Hello_Bike.ipynb
└── requirements.txt
└── url.txt
```

## 2. Project Work Cycles

The project was organized into several key phases to ensure a comprehensive analysis and effective presentation of findings:

### 1. Data Wrangling
- **Gathering Data:** Collected the relevant bike sharing dataset to facilitate analysis.
- **Assessing Data:** Reviewed the dataset for completeness, accuracy, and structure to understand its usability.
- **Cleaning Data:** Addressed any inconsistencies, missing values, and errors in the dataset to prepare it for analysis.

### 2. Exploratory Data Analysis (EDA)
- **Defined Business Questions:** Formulated specific questions to guide the exploration of the data.
- **Create Data Exploration:** Conducted thorough exploratory analysis to uncover insights and patterns in the data that relate to the business questions.

### 3. Data Visualization
- **Create Data Visualization:** Developed visualizations to effectively communicate findings and answer the defined business questions, providing a clearer understanding of the data.

### 4. Dashboard
- **Set Up the DataFrame:** Configured the DataFrame that would be used for the dashboard to ensure it contained all necessary data.
- **Make Filter Components on the Dashboard:** Implemented filter components to allow users to interactively explore the data.
- **Complete the Dashboard with Various Data Visualizations:** Finalized the dashboard by incorporating a variety of visualizations that present the analysis comprehensively.

Here’s a structured outline for the "How to Run This Project" section based on your instructions:

## 3. How to Run this Dashboard

Follow these steps to run the Bike Sharing Dataset analysis project:

1. **Clone this Repository:**
   Open your terminal or command prompt and run the following command to clone the repository:
   ```bash
   git clone https://github.com/SachioTheodore/Data-Analysis-Project-Submission.git
   ```

2. **Install All Required Libraries:**
   Navigate to the project directory and install the necessary libraries using either of the following commands:
   ```bash
   pip install -r requirements.txt
   ```

3. **Go to the Dashboard Folder:**
   Change to the dashboard directory by running:
   ```bash
   cd dashboard
   ```

4. **Run with Streamlit:**
   Start the Streamlit application with the following command:
   ```bash
   streamlit run dashboard/dashboard.py
   ```

5. **Alternatively, Access the Dashboard Online:**
   You can also view the Bike Sharing Dashboard directly by clicking the link below:
   [Click here to view Bike Sharing Dashboard]([your_dashboard_link_here](http://localhost:8502))

