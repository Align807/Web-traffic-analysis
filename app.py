import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Load the dataset
data = pd.read_csv('traffic.csv')

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Filter pageview events
pageviews = data[data['event'] == 'pageview']


# Total pageview events
total_pageviews = pageviews.shape[0]
print(f"Total Pageviews: {total_pageviews}")

# Average pageview events per day
pageviews_per_day = pageviews.groupby('date').size().mean()
print(f"Average Pageviews per Day: {pageviews_per_day}")


# Total count and distribution of events
event_distribution = data['event'].value_counts()
print("Event Distribution:\n", event_distribution)

# Percentage distribution of events
event_percentage = data['event'].value_counts(normalize=True) * 100
print("Event Percentage Distribution:\n", event_percentage)

# Geographical distribution of pageviews by country
geo_distribution = pageviews['country'].value_counts()
print("Geographical Distribution (Top 10 Countries):\n", geo_distribution.head(10))

# Calculate CTR as clicks/pageviews
total_clicks = data[data['event'] == 'click'].shape[0]
CTR = total_clicks / total_pageviews
print(f"Overall CTR: {CTR}")

# CTR per link
link_pageviews = pageviews.groupby('linkid').size()
link_clicks = data[data['event'] == 'click'].groupby('linkid').size()

# Calculate CTR for each link
link_CTR = (link_clicks / link_pageviews).dropna()
print("CTR per link (Top 10):\n", link_CTR.head(10))

# Total clicks and previews per link
link_clicks = data[data['event'] == 'click'].groupby('linkid').size()
link_previews = data[data['event'] == 'preview'].groupby('linkid').size()

# Merge clicks and previews into a single DataFrame for correlation analysis
click_preview_data = pd.DataFrame({
    'clicks': link_clicks,
    'previews': link_previews
}).dropna()

# Perform Pearson and Spearman correlation tests
pearson_corr, pearson_pvalue = pearsonr(click_preview_data['clicks'], click_preview_data['previews'])
spearman_corr, spearman_pvalue = spearmanr(click_preview_data['clicks'], click_preview_data['previews'])

print(f"Pearson Correlation: {pearson_corr}, p-value: {pearson_pvalue}")
print(f"Spearman Correlation: {spearman_corr}, p-value: {spearman_pvalue}")
