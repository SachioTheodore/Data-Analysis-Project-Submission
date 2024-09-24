import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Prepare hour_df
hour_df = pd.read_csv("dashboard/hour.csv")
hour_df.head()

# Delete columns
drop_col = ['instant','yr','atemp']

for i in hour_df.columns:
  if i in drop_col:
    hour_df.drop(labels=i, axis=1, inplace=True)

# Rename columns
hour_df.rename(columns={
    'dteday': 'dateday',
    'mnth': 'month',
    'weathersit': 'weather_cond',
    'cnt': 'count',
    'hr':'hour'
}, inplace=True)

# Change categorical data
hour_df['month'] = hour_df['month'].map({
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
})
hour_df['season'] = hour_df['season'].map({
    1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'
})
hour_df['weekday'] = hour_df['weekday'].map({
    0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'
})
hour_df['weather_cond'] = hour_df['weather_cond'].map({
    1: 'Clear/Partly Cloudy',
    2: 'Misty/Cloudy',
    3: 'Light Snow/Rain',
    4: 'Severe Weather'
})

# Filter Component #################################################################################################################################################################################################################################################
min_date = pd.to_datetime(hour_df['dateday']).dt.date.min()
max_date = pd.to_datetime(hour_df['dateday']).dt.date.max()
 
with st.sidebar:
    st.image('dashboard/logo.gif')
    
    # Take start_date & end_date from date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value= min_date,
        max_value= max_date,
        value=[min_date, max_date]
    )

main_df = hour_df[(hour_df['dateday'] >= str(start_date)) & 
                (hour_df['dateday'] <= str(end_date))]

# Title #################################################################################################################################################################################################################################################
st.markdown("""
    <h1 style='text-align: center;'>
        🚲 Bike Rental Dashboard 💨 <br> 
        <span style='font-size: 20px;'>Your Adventure Awaits!</span>
    </h1>
""", unsafe_allow_html=True)

# (1) Daily Rentals #################################################################################################################################################################################################################################################

# Helper function
def create_daily_rent_df(df):
    daily_rent_df = df.groupby(by='dateday').agg({
        'count': 'sum'
    }).reset_index()
    return daily_rent_df
daily_rent_df = create_daily_rent_df(main_df)

def create_daily_casual_rent_df(df):
    daily_casual_rent_df = df.groupby(by='dateday').agg({
        'casual': 'sum'
    }).reset_index()
    return daily_casual_rent_df
daily_casual_rent_df = create_daily_casual_rent_df(main_df)

def create_daily_registered_rent_df(df):
    daily_registered_rent_df = df.groupby(by='dateday').agg({
        'registered': 'sum'
    }).reset_index()
    return daily_registered_rent_df
daily_registered_rent_df = create_daily_registered_rent_df(main_df)

# Data Visualization
st.subheader('Daily Rentals')
col1, col2, col3 = st.columns(3)

with col1:
    daily_rent_casual = daily_casual_rent_df['casual'].sum()
    st.metric('Casual User', value=f'{daily_rent_casual:,}')

with col2:
    daily_rent_registered = daily_registered_rent_df['registered'].sum()
    st.metric('Registered User', value=f'{daily_rent_registered:,}')

with col3:
    daily_rent_total = daily_rent_df['count'].sum()
    st.metric('Total User', value=f'{daily_rent_total:,}')

# (2) DataFrame #################################################################################################################################################################################################################################################

# Helper function
def create_filtered_df(df, start_date, end_date, weather_condition=None):
    filtered_df = df[(df['dateday'] >= str(start_date)) & (df['dateday'] <= str(end_date))]
    
    if weather_condition:
        filtered_df = filtered_df[filtered_df['weather_cond'] == weather_condition]
    
    return filtered_df

# Data Visualization
st.subheader('Filtered Bike Rentals Table')

min_date = pd.to_datetime(hour_df['dateday']).dt.date.min()
max_date = pd.to_datetime(hour_df['dateday']).dt.date.max()

start_date, end_date = st.date_input(
    label='Select Date Range',
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date]
)

weather_conditions = hour_df['weather_cond'].unique()
selected_weather = st.selectbox('Select Weather Condition', options=['All'] + list(weather_conditions))

# Filter the DataFrame
if selected_weather == 'All':
    filtered_df = create_filtered_df(hour_df, start_date, end_date)
else:
    filtered_df = create_filtered_df(hour_df, start_date, end_date, selected_weather)
st.write(filtered_df)
    
# (3) Monthly Rentals #################################################################################################################################################################################################################################################

# Helper function
def create_monthly_rent_df(df):
    monthly_rent_df = df.groupby(by='month').agg({
        'count': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    })
    ordered_months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    monthly_rent_df = monthly_rent_df.reindex(ordered_months, fill_value=0)
    return monthly_rent_df
monthly_rent_df = create_monthly_rent_df(main_df)

# Data Visualization
st.subheader('Monthly Rentals')

fig, ax = plt.subplots(figsize=(24, 8))
ax.plot(
    monthly_rent_df.index,
    monthly_rent_df['count'],
    marker='o', 
    linewidth=2,
    color='tab:blue',
    label='Total Rentals'
)
ax.plot(
    monthly_rent_df.index,
    monthly_rent_df['casual'],
    marker='o', 
    linewidth=2,
    color='tab:orange',
    label='Casual Rentals'
)
ax.plot(
    monthly_rent_df.index,
    monthly_rent_df['registered'],
    marker='o', 
    linewidth=2,
    color='tab:green',
    label='Registered Rentals'
)
for index, row in monthly_rent_df.iterrows():
    ax.text(index, row['count'] + 1, str(row['count']), ha='center', va='bottom', fontsize=12, color='blue')
    ax.text(index, row['casual'] + 1, str(row['casual']), ha='center', va='bottom', fontsize=12, color='orange')
    ax.text(index, row['registered'] + 1, str(row['registered']), ha='center', va='bottom', fontsize=12, color='green')
ax.tick_params(axis='x', labelsize=25, rotation=45)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=20)
st.pyplot(fig)

# (4) Season and Weather Condition #################################################################################################################################################################################################################################################

# Helper function
def create_season_weather_rent_df(df):
    workdays = df[df['workingday'] == 1]
    non_workdays = df[df['workingday'] == 0]
    return workdays, non_workdays
workdays, non_workdays = create_season_weather_rent_df(main_df)

# Data Visualization
st.subheader('Season and Weather Condition')
plt.figure(figsize=(16, 6))

# Workdays
plt.subplot(1, 2, 1)
sns.barplot(x='season', y='count', hue='weather_cond', data=workdays, palette='Set2')
plt.title('Total Bike Users on Workdays', fontsize=16)
plt.ylabel('Total Users', fontsize=14)
plt.xlabel('Season', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Weather Condition', fontsize=12, title_fontsize='13')

# Non-workdays
plt.subplot(1, 2, 2)
sns.barplot(x='season', y='count', hue='weather_cond', data=non_workdays, palette='Set2')
plt.title('Total Bike Users on Non-Workdays', fontsize=16)
plt.ylabel('Total Users', fontsize=14)
plt.xlabel('Season', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Weather Condition', fontsize=12, title_fontsize='13')

plt.tight_layout()
st.pyplot(plt)

# (5) Weather Parameters #################################################################################################################################################################################################################################################

# Helper function
def create_monthly_grouped_df(df):
    monthly_grouped = df.groupby(['month']).agg({
        'casual': 'sum',
        'registered': 'sum',
        'temp': 'mean',
        'hum': 'mean',
        'windspeed': 'mean'
    }).reset_index()

    monthly_grouped['normalized_temp'] = monthly_grouped['temp']
    monthly_grouped['normalized_humidity'] = monthly_grouped['hum']
    monthly_grouped['normalized_wind_speed'] = monthly_grouped['windspeed']

    return monthly_grouped
monthly_grouped = create_monthly_grouped_df(main_df)

# Data Visualization
st.subheader('Effect of Weather Parameters on Bike Users')
plt.figure(figsize=(15, 5))

# Scatter plot for temperature vs users
plt.subplot(1, 3, 1)
sns.scatterplot(data=monthly_grouped, x='normalized_temp', y='casual', label='Casual Users', color='blue', alpha=0.6)
sns.scatterplot(data=monthly_grouped, x='normalized_temp', y='registered', label='Registered Users', color='orange', alpha=0.6)
sns.regplot(data=monthly_grouped, x='normalized_temp', y='casual', scatter=False, color='blue', line_kws={'linestyle':'--'})
sns.regplot(data=monthly_grouped, x='normalized_temp', y='registered', scatter=False, color='orange', line_kws={'linestyle':'--'})
plt.title('Effect of Temperature on Bike Users')
plt.xlabel('Normalized Temperature')
plt.ylabel('Number of Users')
plt.grid(True)
plt.legend()

# Scatter plot for humidity vs users
plt.subplot(1, 3, 2)
sns.scatterplot(data=monthly_grouped, x='normalized_humidity', y='casual', label='Casual Users', color='blue', alpha=0.6)
sns.scatterplot(data=monthly_grouped, x='normalized_humidity', y='registered', label='Registered Users', color='orange', alpha=0.6)
sns.regplot(data=monthly_grouped, x='normalized_humidity', y='casual', scatter=False, color='blue', line_kws={'linestyle':'--'})
sns.regplot(data=monthly_grouped, x='normalized_humidity', y='registered', scatter=False, color='orange', line_kws={'linestyle':'--'})
plt.title('Effect of Humidity on Bike Users')
plt.xlabel('Normalized Humidity')
plt.ylabel('')
plt.grid(True)
plt.legend()

# Scatter plot for wind speed vs users
plt.subplot(1, 3, 3)
sns.scatterplot(data=monthly_grouped, x='normalized_wind_speed', y='casual', label='Casual Users', color='blue', alpha=0.6)
sns.scatterplot(data=monthly_grouped, x='normalized_wind_speed', y='registered', label='Registered Users', color='orange', alpha=0.6)
sns.regplot(data=monthly_grouped, x='normalized_wind_speed', y='casual', scatter=False, color='blue', line_kws={'linestyle':'--'})
sns.regplot(data=monthly_grouped, x='normalized_wind_speed', y='registered', scatter=False, color='orange', line_kws={'linestyle':'--'})
plt.title('Effect of Wind Speed on Bike Users')
plt.xlabel('Normalized Wind Speed')
plt.ylabel('')
plt.grid(True)
plt.legend()

plt.tight_layout()
st.pyplot(plt)

# (6) K-Means #################################################################################################################################################################################################################################################

# Helper function (calculate inertia values for elbow method)
def calculate_inertia(features, max_clusters=20):
    inertia_values = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(features)
        inertia_values.append(kmeans.inertia_)
    return inertia_values

# Helper function (perform KMeans clustering)
def perform_kmeans(features, k):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(features)
    return clusters, kmeans.cluster_centers_

# Data preparation
features = hour_df[['temp', 'hum', 'count']]  # Menggunakan kolom yang sudah diubah
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Sidebar input for k value
st.sidebar.subheader('Input k for K-Means Clustering')
k_value = st.sidebar.slider('Select number of clusters (k)', min_value=1, max_value=10, value=3)

st.subheader('K-Means Clustering Analysis:')
st.write('#### Elbow Method for Optimal k')

# Calculate inertia values and plot elbow method
inertia_values = calculate_inertia(scaled_features)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, 21), inertia_values, marker='o')
ax.set_title('Elbow Method for Optimal k', fontsize=16)
ax.set_xlabel('Number of Clusters (k)', fontsize=14)
ax.set_ylabel('Inertia', fontsize=14)
ax.grid(True)
st.pyplot(fig)

st.write(f'#### K-Means Clustering Results (k = {k_value})')

# Perform KMeans clustering with the selected k
hour_df['cluster'], centroids = perform_kmeans(scaled_features, k_value)

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    x=scaled_features[:, 0], 
    y=scaled_features[:, 1], 
    hue=hour_df['cluster'], 
    palette='Set2', alpha=0.6, s=100, ax=ax
)
ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='X')
ax.set_title('K-Means Clustering Results (Normalized)', fontsize=16)
ax.set_xlabel('Normalized Temperature', fontsize=14)
ax.set_ylabel('Normalized Humidity', fontsize=14)
ax.legend()
ax.grid(True)
st.pyplot(fig)
