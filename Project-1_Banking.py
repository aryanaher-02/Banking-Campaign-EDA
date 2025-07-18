import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Coding\DsResearch\Banking\banking_data.csv")

pd.set_option('display.max_columns', None)

print(df.columns)
print(df.isna().sum())
print(df['age'].describe())

print(df['y'].value_counts(dropna=False))

sns.histplot(df['age'], kde=True) 
plt.show()

print(df[df['age'] >= 90].shape[0])

#Age Group Segmentation
df['age_group'] = pd.cut(df['age'], bins = [18, 30, 45, 60, 75, 100], 
                         labels=['Young (18-29)','Adult (30-44)','Mid-Life (45-59)','Senior (60-74)','Elderly (75+)'], 
                         ordered=True)

print(df['age_group'].head())

#Group by age_group and calculate subscription rates
subscription_rates = df.groupby('age_group')['y'].apply(lambda x: (x=='yes').mean() * 100).reset_index(name='subscription_rate')
# % of "yes" per group

#Sort by age_group
subscription_rates = subscription_rates.sort_values('age_group')

print(subscription_rates)

#age distribution
plt.figure(figsize=(12,6))
sns.histplot(data=df, x='age', bins=20, kde=True, color='#1f77b4')
plt.title('Client Age Distribution', pad=20)
plt.xlabel('Age (years)')
plt.ylabel('Number of Clients')
plt.axvline(x=39, color='red', linestyle='--', label='Median (39 years)')
plt.legend()
plt.show()

#boxplot
plt.figure(figsize=(8,4))
sns.boxplot(x=df['age'], color='#ff7f0e')
plt.title('Age Distribution Spread')
plt.xlabel('Age')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(data=subscription_rates, x='age_group', y='subscription_rate', palette='viridis')
plt.title('Subscription Rate by Age Group')
plt.show()

print(df['job'].value_counts(dropna = False))

df['job'] = df['job'].replace('unknown', df['job'].mode()[0]) # Replaced with mode
print(df['job'].value_counts(dropna = False))
job_counts = df['job'].value_counts()

plt.figure(figsize=(12,6))
job_plot = sns.barplot(x=job_counts.index, y=job_counts.values, palette='viridis')

plt.title('Client Distribution by Job Type', fontsize=16)
plt.xlabel('Job Category', fontsize=12)
plt.ylabel('Number of Clients', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate subscription rates by job
job_subscription = df.groupby('job')['y'].apply(lambda x: (x=='yes').mean() * 100).sort_values(ascending=False)

print(job_subscription)

plt.figure(figsize=(12,6))
sns.barplot(x=job_subscription.index, y=job_subscription.values, palette='coolwarm')
plt.title('Subscription Rate by Job Type', fontsize=16)
plt.xlabel('Job Category', fontsize=12)
plt.ylabel('Subscription Rate (%)', fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 60)
plt.tight_layout()
plt.show()

print(df['marital'].value_counts(dropna = False))
print(df['marital_status'].value_counts(dropna = False))
#dropping one column
df.drop('marital', axis=1, inplace=True)
#mode imputation for the 3 missing values
df['marital_status'].fillna('married', inplace=True)

# Calculate counts and percentages
marital_counts = df['marital_status'].value_counts()
marital_percent = df['marital_status'].value_counts(normalize=True) * 100

# Display
print("Absolute Counts:")
print(marital_counts.to_string())
print("\nPercentages:")
print(marital_percent.round(1).to_string())

plt.figure(figsize=(10,6))
sns.barplot(x=marital_counts.index,  y=marital_counts.values, palette=['#3498db', '#e74c3c', '#2ecc71'],
            order=['married', 'single', 'divorced'])

plt.title('Client Distribution by Marital Status', fontsize=16, pad=20)
plt.xlabel('Marital Status', fontsize=12)
plt.ylabel('Number of Clients', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Calculating subscription rates
marital_subs = df.groupby('marital_status')['y'].apply(lambda x: (x == 'yes').mean() *100).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=marital_subs.index, y=marital_subs.values, 
            palette=['#2ecc71', '#3498db', '#e74c3c'], order=marital_subs.index)

plt.title('Subscription Rate by Marital Status', fontsize=16, pad=20)
plt.xlabel('Marital Status', fontsize=12)
plt.ylabel('Subscription Rate (%)', fontsize=12)
plt.ylim(0, 20)
plt.tight_layout()
plt.show()

print("\nSubscription rate by Marital Status:")
print(marital_subs)

#trying pie chart
data = {'married':60.2, 'single':28.3, 'divorced':11.5}
colors = ['#3498db','#e74c3c','#2ecc71']  # Blue, Red, Green

plt.figure(figsize=(8,8))
plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Client Marital Status Distribution', fontsize=16, pad=20)
plt.savefig('marital_pie.png', dpi=300, bbox_inches='tight')
plt.show()

print(df['education'].value_counts(dropna = False))

#mode imputation
df['education'] = df['education'].replace(['unknown', np.nan], df['education'].mode()[0])

print(df['education'].value_counts(dropna = False))
print(df['education'].value_counts(normalize=True)*100)

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='education', order=['primary','secondary','tertiary'], palette='Blues_r')
plt.title('Client Education Level Distribution', fontsize=14, pad=10)
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

print(df['y'].value_counts(dropna = False))

df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Subscription Performance Analysis
plt.figure(figsize=(10,6))
edu_subs = df.groupby('education')['y'].mean().sort_values(ascending=False)
sns.barplot(x=edu_subs.index, y=edu_subs.values*100, 
            order=['tertiary','secondary','primary'], palette=['#27ae60','#3498db','#e67e22'])
plt.title('Subscription Rate by Education Level', fontsize=16, pad=20)
plt.xlabel('')
plt.ylabel('Conversion Rate (%)', fontsize=12)
plt.ylim(0, 25)
plt.show()

data = {'education': ['tertiary', 'secondary', 'primary'], 'conversion_rate': [0.150139, 0.107924, 0.086411]}

edu_subs_pct = (edu_subs * 100).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=edu_subs_pct.index,y=edu_subs_pct.values,order=['tertiary','secondary','primary'],
            palette=['#2ecc71','#3498db','#e67e22'], saturation=0.85)

# Calculating overall average for reference line
overall_avg = (df['y'] == 1).mean() * 100

plt.title('TERM DEPOSIT CONVERSION BY EDUCATION LEVEL\n', fontsize=16, pad=20, weight='bold')
plt.xlabel('')
plt.ylabel('Conversion Rate (%)', fontsize=12)
plt.ylim(0, max(edu_subs_pct) + 5)
plt.axhline(y=overall_avg, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
plt.text(0.5, overall_avg + 1, f'Overall Average = {overall_avg:.1f}%', 
         ha='center', va='center', color='red',
         bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'))
plt.tight_layout()
plt.savefig('auto_education_conversion.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nConversion Rates:")
print(edu_subs_pct.to_frame('Conversion Rate (%)'))

# print(df['default'].value_counts(dropna = False))
# Renamed the column for clarity
df = df.rename(columns={'default': 'credit_in_default'})

default_stats = df['credit_in_default'].value_counts(normalize=True).mul(100).round(2)

print("Credit Default Status:")
print(default_stats.to_string())

print(df['credit_in_default'].value_counts(dropna = False))

plt.figure(figsize=(8,6))
default_stats.plot(kind='bar', color=['#2ecc71','#e74c3c'], rot=0)
 # Green for 'no', red for 'yes'
plt.title('Client Credit Default Status', pad=20)
plt.ylabel('Percentage of Clients', labelpad=10)
plt.xlabel('Credit in Default', labelpad=10)
plt.ylim(0, 100)
plt.show()

print(df['balance'].value_counts(dropna = False))
print(df['balance'].describe())
print(df['balance'].isna().sum())

# Checking how many negative balances exist
negative_count = (df['balance'] < 0).sum()
negative_pct = negative_count / len(df) * 100

print(f"Negative balances: {negative_count} clients ({negative_pct:.2f}%)")

df['balance_clean'] = df['balance'].clip(lower=0)
df['overdraft_flag'] = df['balance'] < 0
df['overdraft_amount'] = df['balance'].where(df['balance'] < 0, 0).abs()

print(df['overdraft_amount'].describe().apply(lambda x: f"{x:,.2f}"))

# Clients with any overdraft
print(f"Clients with overdrafts: {(df['overdraft_amount'] > 0).sum()} ({(df['overdraft_amount'] > 0).mean()*100:.2f}%)")

# Severe overdrafts (>€1000)
severe_overdrafts = df[df['overdraft_amount'] > 1000]
print(f"\nSevere overdrafts (>€1000): {len(severe_overdrafts)} clients")
print(severe_overdrafts['overdraft_amount'].describe().apply(lambda x: f"{x:,.2f}"))

# Create risk tiers
df['risk_tier'] = pd.cut(df['overdraft_amount'], bins=[0, 100, 1000,float('inf')],
                         labels=['low', 'medium','high'], right=False)

# Tier distribution
tier_dist=df['risk_tier'].value_counts(normalize=True).mul(100).round(2)
print(tier_dist)

# Main Balance Distribution Plot
plt.figure(figsize=(12, 6))
sns.histplot(df['balance'], bins=100, kde=True, color='#3498db')
plt.axvline(df['balance'].median(), color='#e74c3c', linestyle='--', 
            label=f'Median: €{df["balance"].median():,.0f}')
plt.axvline(df['balance'].mean(), color='#2ecc71', linestyle='--', 
            label=f'Mean: €{df["balance"].mean():,.0f}')
plt.xlim(-5000, 20000)
plt.title('Client Balance Distribution', fontsize=16)
plt.xlabel('Balance (€)', fontsize=12)
plt.ylabel('Number of Clients', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# Overdraft Concentration Plot
plt.figure(figsize=(10, 6))
overdrafts = df[df['overdraft_amount'] > 0]['overdraft_amount']
sns.histplot(overdrafts, bins=30, color='#e74c3c')
plt.axvline(1000, color='black', linestyle=':', label='High-risk threshold (€1k)')
plt.title('Overdraft Amount Distribution (144 clients > €1k)', fontsize=16)
plt.xlabel('Overdraft Amount (€)', fontsize=12)
plt.ylabel('Client Count', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='y', y='balance', palette=['#e74c3c','#2ecc71'],showfliers=False)
plt.ylim(-1000, 5000)
plt.title('Balance Distribution by Subscription Status\n(Subscribers have higher balances)', fontsize=16)
plt.xlabel('Subscribed to Term Deposit?', fontsize=12)
plt.ylabel('Balance (€)', fontsize=12)
plt.tight_layout()
plt.show()

# Balance statistics by subscription status
balance_stats = df.groupby('y')['balance'].describe().applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

print("Balance Distribution by Subscription Status:")
print(balance_stats[['count', 'mean', '50%', 'min', 'max']])
# print(balance_stats)

df['housing'] = df['housing'].map({'yes': 1, 'no': 0}).astype(int)

print(df['housing'].value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(x='housing', data=df, order=[0, 1], palette=['#3498db', '#e74c3c'])
plt.title('Number of Clients with Housing Loans')
plt.xlabel('Housing Loan')
plt.ylabel('Count')
plt.xticks(ticks = [0, 1], labels = ['No', 'Yes'])
plt.show()

# pie chart
plt.figure(figsize=(8, 6))
housing_counts = df['housing'].value_counts()
plt.pie(housing_counts, labels=['Yes', 'No'], autopct='%1.1f%%', startangle=90,
        colors=['lightgreen', 'lightcoral'])
plt.title('Proportion of Clients with Housing Loans')

# Housing Loan Status vs Term Deposit Subscription
plt.figure(figsize=(8, 6))
sns.countplot(x='housing', hue='y', data=df)
plt.title('Housing Loan Status vs Term Deposit Subscription')
plt.xlabel('Housing Loan')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend(title='Subscribed Term Deposit', labels=['No', 'Yes'])
plt.show()

#  Statistical Analysis
print("\nHousing Loan Status vs Term Deposit Subscription:")
cross_tab = pd.crosstab(df['housing'], df['y'], margins=True, margins_name="Total")
percentage_tab = pd.crosstab(df['housing'], df['y'], normalize='index').round(4)*100

print("\nAbsolute Counts:")
print(cross_tab)
print("\nPercentage Breakdown:")
print(percentage_tab)

print(df['loan'].value_counts(dropna = False))
df = df.rename(columns={'loan': 'personal_loan'})
df['personal_loan'] = df['personal_loan'].str.title()

sns.countplot(x='personal_loan', data=df, hue='personal_loan', 
              palette={'No': '#4e79a7', 'Yes': '#f28e2b'}, order=['No', 'Yes'])
plt.title('Clients with Personal Loans')
plt.xlabel('Personal Loan')
plt.ylabel('Count')
plt.show()

loan_counts = df['personal_loan'].value_counts()
plt.pie(loan_counts, labels=loan_counts.index, autopct='%1.1f%%', 
        colors=['#4e79a7', '#f28e2b'], startangle=90, 
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
plt.title('Proportion of Personal Loans')
plt.tight_layout()
plt.show()

# Loan holders by job type
plt.figure(figsize=(10, 6))
sns.countplot(y='job', hue='personal_loan', data=df, palette={'No':'#3498db','Yes':'#e74c3c'}, 
              order=df['job'].value_counts().index)
plt.title('Personal Loan Holders by Job Type')
plt.xlabel('Count')
plt.ylabel('Job Category')
plt.legend(title='Personal Loan')
plt.show()

# Age distribution comparison
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='personal_loan', element='step', stat='density', 
             palette={'No':'#3498db','Yes':'#e74c3c'}, common_norm=False)
plt.title('Age Distribution: Loan vs No-Loan Clients')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()
contact_counts = df['contact'].value_counts()
contact_perc = df['contact'].value_counts(normalize=True).mul(100).round(2)

print("Absolute Counts:")
print(contact_counts)
print("\nPercentage Breakdown:")
print(contact_perc)

plt.figure(figsize=(12, 5))
sns.countplot(x='contact', data=df, order=contact_counts.index,
              palette= {'cellular':'#2ecc71', 'telephone':'#e74c3c', 'unknown':'#95a5a6'})
plt.title('Communication Types Used', pad=20)
plt.xlabel('Contact Method')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 8))

colors = ['#2ecc71', '#95a5a6', '#e74c3c']

plt.pie(contact_counts,labels=contact_counts.index,autopct='%1.1f%%',
        startangle=90,colors=colors,shadow=False, 
        textprops={'fontsize': 16, 'color': 'white', 'fontweight': 'bold'})
plt.axis('equal')  
plt.title('Contact Methods Distribution', pad=2, fontsize=14)
plt.legend(contact_counts.index, title="Contact Method", loc="best", 
           fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.show()

print(df['y'].value_counts(dropna = False))

# Subscription counts by contact method
subscription_counts = df[df['y'] == 1]['contact'].value_counts()
print("Subscriptions by Contact Method:")
print(subscription_counts)

# Conversion rates
conversion_rates = df.groupby('contact')['y'].mean() * 100
print("\nConversion Rates (%):")
print(conversion_rates.round(2))

# Plot: Subscriptions by Method (Count)
plt.figure(figsize=(10,5))
sns.countplot(x='contact', data=df[df['y'] == 1], order=['cellular','unknown','telephone'],
              palette=['#4CAF50','#9E9E9E','#F44336'])
plt.title('Term Deposit Subscriptions by Contact Method')
plt.xlabel('Contact Type')
plt.ylabel('Subscription Count')
plt.show()


plt.figure(figsize=(10,5))
sns.barplot(x=conversion_rates.index, y=conversion_rates.values, 
            order=['telephone','cellular','unknown'], 
            palette=['#F44336','#4CAF50','#9E9E9E'])
plt.title('Conversion Rates by Contact Method')
plt.xlabel('Contact Type')
plt.ylabel('Conversion Rate (%)')
plt.ylim(0, 18)
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x='contact', hue='y', data=df, order=['cellular','unknown','telephone'], 
              palette={0:'#F44336', 1:'#4CAF50'})
plt.title('Contact Method vs Subscription Outcome')
plt.xlabel('Contact Type')
plt.ylabel('Count')
plt.legend(title='Subscribed?', labels=['No','Yes'])
plt.show()

day_counts = df['day'].value_counts().sort_index()
print("\nDay-wise Counts:")
print(day_counts)

sns.set_style("whitegrid")
plt.figure(figsize=(15, 6))

# Histogram with KDE
plt.subplot(1, 2, 1)
sns.histplot(df['day'], bins=31, kde=True, color='#3498db')
plt.title('Distribution of Contact Days')
plt.xlabel('Day of Month')
plt.ylabel('Number of Contacts')
plt.axvline(df['day'].mean(), color='red', linestyle='--', label=f'Mean: {df["day"].mean():.1f}')
plt.legend()

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=df['day'], color='#2ecc71')
plt.title('Spread of Contact Days')
plt.xlabel('Day of Month')

plt.tight_layout()
plt.show()

# Line plot for trend analysis
plt.figure(figsize=(12, 6))
sns.lineplot(x=day_counts.index, y=day_counts.values, marker='o', color='#e74c3c')
plt.title('Contact Frequency by Day of Month')
plt.xlabel('Day of Month')
plt.ylabel('Number of Contacts')
plt.xticks(range(1, 32))
plt.grid(True)
plt.show()

# Week analysis (assuming 1-7 = week 1, 8-14 = week 2, etc.)
df['week_of_month'] = pd.cut(df['day'], 
                            bins=[0, 7, 14, 21, 28, 32],
                            labels=['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5'])

# Week-wise contacts
week_counts = df['week_of_month'].value_counts()
print("\nContacts by Week of Month:")
print(week_counts)

# Week-wise conversion rates
week_conversion = df.groupby('week_of_month')['y'].mean().sort_values(ascending=False)
print("\nConversion Rates by Week:")
print(week_conversion)


#orderwise counts for last contact month
month_counts = df['month'].value_counts().reindex(['jan', 'feb', 'mar', 'apr', 
  'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
print("Contact Counts by Month:")
print(month_counts)

# Percentage breakdown
print("\nPercentage Distribution:")
print((month_counts / len(df) * 100).round(2))

sns.set_style("whitegrid")
plt.figure(figsize=(15, 6))

# Bar plot
plt.subplot(1, 2, 1)
sns.countplot(x='month', data=df, 
              order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 
              palette='viridis')
plt.title('Contact Volume by Month')
plt.xlabel('Month')
plt.ylabel('Number of Contacts')
plt.xticks(rotation=45)

# Pie chart
plt.subplot(1, 2, 2)
month_counts.plot.pie(autopct='%1.1f%%', startangle=90,colors=sns.color_palette('viridis', 12), 
                      wedgeprops={'edgecolor':'white'})
plt.title('Monthly Contact Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

df["duration"] = df["duration"].where(df["duration"] > 0)
print("Duration Statistics (seconds):")
print(df['duration'].describe().round(2))

duration = df["duration"].dropna()

sns.set_style("whitegrid")
plt.figure(figsize=(18, 6))

# Histogram with KDE
plt.subplot(1, 3, 1)
sns.histplot(duration, bins=50, kde=True, color='#3498db')
plt.title('Call Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.axvline(duration.mean(), color='red', linestyle='--', label=f'Mean: {duration.mean():.1f}s')
plt.legend()

# Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(x=duration, color='#2ecc71')
plt.title('Call Duration Spread')
plt.xlabel('Seconds')

# Cumulative distribution
plt.subplot(1, 3, 3)
sns.ecdfplot(duration, color='#e74c3c')
plt.title('Cumulative Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.ylabel('Proportion of Calls')
plt.axhline(0.5, color='grey', linestyle='--')
plt.axvline(180, color='grey', linestyle='--', label='Median (180s)')
plt.legend()
plt.show()

print(df['campaign'].describe())
counts = df['campaign'].value_counts().sort_index()
print(counts)
print(counts.sum())
print(counts[counts.index > 2].sum())


plt.figure(figsize=(12, 6))
ax = sns.histplot(data=df, x='campaign', bins=30, kde=False, color='#3498db')
plt.yscale('log')  # Log scale for better visibility
plt.title('Distribution of Campaign Contacts (Log Scale)', fontsize=14, pad=20)
plt.xlabel('Number of Contacts', fontsize=12)
plt.ylabel('Log(Count of Clients)', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Annotate key points
plt.axvline(x=2, color='red', linestyle='--', alpha=0.7, label='Median (2 contacts)')
plt.axvline(x=3, color='orange', linestyle='--', alpha=0.7, label='75th Percentile (3 contacts)')
plt.legend()
plt.show()


contacts = counts.index
frequency = counts.values
cumulative_percentage = np.cumsum(frequency) / 45216 * 100

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot (frequency)
ax1.bar(contacts[:15], frequency[:15], color='#3498db')
ax1.set_xlabel('Number of Contacts', fontsize=12)
ax1.set_ylabel('Number of Clients', color='#3498db', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#3498db')
ax1.set_xticks(range(1, 16))

# Line plot (cumulative %)
ax2 = ax1.twinx()
ax2.plot(contacts[:15], cumulative_percentage[:15], color='#e74c3c', marker='o', linewidth=2)
ax2.set_ylabel('Cumulative Percentage (%)', color='#e74c3c', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax2.set_ylim(0, 100)

plt.title('Pareto Analysis of Campaign Contacts (First 15 Values)', fontsize=14, pad=20)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Calculate subscription rate (y=1) per contact attempt
subscription_rate = df.groupby('campaign')['y'].mean() * 100  # Convert to percentage

# Filter to focus on typical range (1-15 contacts)
plot_data = subscription_rate[subscription_rate.index <= 15]

plt.figure(figsize=(12, 6))
ax = sns.lineplot(x=plot_data.index, y=plot_data.values, marker='o', 
                  markersize=8, color='#3498db',linewidth=2.5)

# Highlight key points
max_rate = plot_data.max()
max_contacts = plot_data.idxmax()
plt.axvline(x=max_contacts, color='#e74c3c', linestyle='--', alpha=0.7, 
            label=f'Peak: {max_rate:.1f}% at {max_contacts} contacts')

# Annotate inflection points
for x in [1, max_contacts, 5, 10]:
    if x in plot_data.index:
        y = plot_data.loc[x]
        ax.annotate(f'{y:.1f}%', xy=(x, y),xytext=(5, 5), textcoords='offset points', 
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

plt.title('Term Deposit Subscription Rate by Number of Contacts', fontsize=14, pad=20)
plt.xlabel('Number of Contacts During Campaign', fontsize=12)
plt.ylabel('Subscription Rate (%)', fontsize=12)
plt.xticks(range(1, 16))
plt.ylim(0, plot_data.max() * 1.1)  # Add 10% headroom
plt.grid(axis='y', alpha=0.3)
plt.legend()
sns.despine()
plt.show()

plt.figure(figsize=(8, 5))

# boxplot
sns.boxplot(y=df['campaign'], color='skyblue', width=0.4, showmeans=True, 
            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'red'})

# essential annotations
plt.title('Distribution of Contacts Per Client', fontsize=14, pad=15)
plt.ylabel('Number of Contacts', fontsize=12)
plt.yticks(fontsize=10)

sns.despine()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Set clipping threshold (adjust as needed)
CLIP_THRESHOLD = 10  # Will focus on 0-10 contacts range

plt.figure(figsize=(8, 5))

# Create clipped boxplot
sns.boxplot(y=df['campaign'].clip(upper=CLIP_THRESHOLD), color='#1f77b4', width=0.4, 
            showfliers=False, showmeans=True, 
            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'red'})

plt.text(0.5, CLIP_THRESHOLD-0.5, 
         f'Values > {CLIP_THRESHOLD} clipped\n({len(df[df["campaign"]>CLIP_THRESHOLD])} clients affected)',
ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.title(f'Contacts Distribution (Clipped at {CLIP_THRESHOLD})', fontsize=14, pad=15)
plt.ylabel('Number of Contacts', fontsize=12)
plt.yticks(range(0, CLIP_THRESHOLD+1))
plt.grid(axis='y', alpha=0.2)
sns.despine()

plt.tight_layout()
plt.show()
#for visual clarity, we can add a rolling average (3-contact window) to smooth noise:
plt.figure(figsize=(12, 6))
rolling_avg = plot_data.rolling(window=3, center=True).mean()

sns.lineplot(x=plot_data.index, y=plot_data.values, color='#3498db', alpha=0.3, label='Actual')
sns.lineplot(x=rolling_avg.index, y=rolling_avg.values, color='#e74c3c', 
             linewidth=2.5, label='3-Contact Rolling Avg')

plt.title('Smoothed Subscription Rate Trend', fontsize=14, pad=20)
plt.xlabel('Number of Contacts')
plt.ylabel('Subscription Rate (%)')
plt.legend()
plt.show()

# Define contact ranges
bins = [0, 3, 5, 10, float('inf')]
labels = ['1-3', '4-5', '6-10', '>10']

# Calculate percentage of clients in each range
df['contact_range'] = pd.cut(df['campaign'], bins=bins, labels=labels)
strat_implications = df['contact_range'].value_counts(normalize=True).sort_index() * 100

# Create actionable recommendations
recommendations = {
    '1-3': "Optimal range for conversions",
    '4-5': "Diminishing returns",
    '6-10': "Likely inefficient",
    '>10': "Audit for waste"
}

# Build the strategic table
strat_table = pd.DataFrame({'Contact Range': strat_implications.index,'% of Clients': 
                            strat_implications.values.round(1),'Recommended Action': [recommendations[x] for x in strat_implications.index]})

print("\nStrategic Implications:")
print(strat_table.to_string())  

#working on pdays
df['pdays_clean'] = df['pdays'].replace(-1, np.nan)
print(df['pdays_clean'].describe(percentiles=[.25, .5, .75, .95, .99]).round(2))

contacted = df[df['pdays'] != -1].copy()

# Client Count by Days Since Last Contact
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Create the plot
ax = sns.histplot( data=contacted, x='pdays', bins=30, color='#1f77b4', kde=False)

# Add annotations
plt.title('Client Count by Days Since Last Contact', fontsize=16, pad=20)
plt.xlabel('Days Since Last Contact', fontsize=12)
plt.ylabel('Number of Clients', fontsize=12)

# Highlight key ranges
plt.axvspan(0, 30, color='green', alpha=0.1, label='<1 Month')
plt.axvspan(30, 90, color='orange', alpha=0.1, label='1-3 Months')
plt.axvspan(90, 180, color='red', alpha=0.1, label='3-6 Months')

plt.legend(title='Key Periods:')
plt.grid(axis='y', alpha=0.3)
sns.despine()

plt.tight_layout()
plt.show()

# Preparing data (excluding never-contacted)
contact_days = df[df['pdays'] != -1]['pdays'].value_counts().sort_index()

# Create plot
plt.figure(figsize=(15,6))

# Main plot (0-90 days)
ax1 = plt.subplot(1,2,1)
contact_days[contact_days.index <= 90].plot( marker='o', markersize=5, color='#1f77b4')
plt.title('Daily Contact Patterns (0-90 days)', pad=15)
plt.xlabel('Days Since Last Contact')
plt.ylabel('Number of Clients')
plt.grid(alpha=0.3)

# Highlight weekly spikes
for day in [7,14,21,28,30,60,90]:
    if day in contact_days.index:
        plt.axvline(day, color='grey', linestyle=':', alpha=0.5)

# Long-tail plot (91-871 days)
ax2 = plt.subplot(1,2,2)
contact_days[contact_days.index > 90].plot(marker='o', markersize=3, color='#ff7f0e')
plt.title('Long-Tail Contacts (91-871 days)', pad=15)
plt.xlabel('Days Since Last Contact')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Contact recency segments
bins = [0, 30, 90, 180, 365, float('inf')]
labels = ['<1 Month', '1-3 Months', '3-6 Months', '6-12 Months', '>1 Year']
contacted['recency'] = pd.cut(df['pdays'], bins=bins, labels=labels)

# Plot
plt.figure(figsize=(10, 6))
recency_counts = contacted['recency'].value_counts().sort_index()
recency_counts.plot(kind='bar', color='teal')
plt.title('Client Contact Recency Segments')
plt.xlabel('Time Since Last Contact')
plt.ylabel('Number of Clients')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['pdays_clean'])
plt.title('Days Since Last Contact')
plt.ylabel('Days')
plt.show()

# Bar plot with conversion rates
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='recency', y='y', data=contacted, order=labels,palette="viridis", 
                 estimator=lambda x: np.mean(x)*100, ci=None)

# Adding annotations
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5),  textcoords='offset points',fontsize=10)

plt.title('Subscription Rate by Contact Recency', fontsize=14, pad=20)
plt.xlabel('Time Since Last Contact', fontsize=12)
plt.ylabel('Subscription Rate (%)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
sns.despine()
plt.show()


#previous
print(df['previous'].describe(percentiles=[.25, .5, .75, .90, .95, .99]))
print(df['previous'].value_counts(dropna = False).sort_index())

# Extreme Outlier (275 contacts)
df_clean = df[df['previous'] != 275]
print(df_clean['previous'].value_counts().sort_index())
print(df_clean['previous'].describe(percentiles=[.25, .5, .75, .90, .95, .99]).round(2))

plt.figure(figsize=(14,6))
ax = sns.barplot(x=df_clean['previous'].value_counts().index[:21],  
                 y=df_clean['previous'].value_counts().values[:21], palette="Blues_r")

# Annotate bars
for i, (val, count) in enumerate(df_clean['previous'].value_counts().sort_index().head(21).items()):
    if count > 50:
        ax.text(i, count+500, f'{count:,}', ha='center')

plt.title('Contact Attempt Distribution (0-20 attempts)', pad=20)
plt.xlabel('Number of Previous Contacts')
plt.ylabel('Client Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.2)
plt.ylim(0, df_clean['previous'].value_counts().max()*1.1)
plt.show()

# Filter data (1-20 contacts)
counts = df_clean['previous'].value_counts().sort_index()[1:21]

# Create plot
plt.figure(figsize=(14, 6))
bars = plt.bar(counts.index, counts.values, color='#1f77b4', width=0.8)

# Annotating bars
for bar in bars:
    height = bar.get_height()
    if height >= 10:  # Only label meaningful bars
        plt.text(bar.get_x() + bar.get_width()/2, height+5, f'{int(height)}',
                 ha='center' ,va='bottom' ,fontsize=9)

plt.title('Distribution of Previous Contacts (1-20 attempts)', pad=20)
plt.xlabel('Number of Previous Contacts')
plt.ylabel('Number of Clients')
plt.xticks(range(1, 21, 2))
plt.grid(axis='y', alpha=0.2)
plt.annotate('Note: 36,956 clients had 0 contacts (excluded)', xy=(11, counts.max()*0.6), ha='center', 
             fontstyle='italic')
plt.show()

# Pareto analysis 

contacted = df_clean[df_clean['previous'] > 0]

counts = contacted['previous'].value_counts().sort_index()
cumulative_pct = counts.cumsum() / counts.sum() * 100

fig, ax1 = plt.subplots(figsize=(10, 5))

counts.plot(kind='bar', color='skyblue', ax=ax1)
ax1.set_ylabel('Number of Clients', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

ax2 = ax1.twinx()
cumulative_pct.plot(color='orange', marker='o', ax=ax2)
ax2.set_ylabel('Cumulative Percentage', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.axhline(80, color='red', linestyle='--')

plt.title('Pareto Analysis of Previous Contacts (Excluding 0)')
ax1.set_xlabel('Number of Previous Contacts')
plt.grid(False)
plt.tight_layout()
plt.show()


contacted = df_clean.copy()
plt.figure(figsize=(12, 6))

conversion_rates = contacted.groupby('previous')['y'].mean().reset_index()

# Plot with 95% confidence intervals
ax = sns.barplot(data=contacted[contacted['previous'] <= 10], x='previous', y='y', 
                 palette='viridis', ci = None)

# Annotate bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.title('Subscription Rate by Number of Previous Contacts', fontsize=14)
plt.xlabel('Previous Contact Attempts')
plt.ylabel('Subscription Rate')
plt.ylim(0, 0.4)
sns.despine()
plt.show()

print(conversion_rates.value_counts().sort_index())
print(conversion_rates.describe())

#poutcome
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")


ax = sns.countplot(data=df, x='poutcome', order=['unknown', 'failure', 'other', 'success'],
                   palette=['#999999', '#e74c3c', '#3498db', '#2ecc71'])

# Annotations
for p in ax.patches:
    ax.annotate(f'{p.get_height():,}\n({p.get_height()/len(df)*100:.1f}%)',
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.title("Previous Campaign Outcomes Distribution", pad=20)
plt.xlabel("Outcome")
plt.ylabel("Count")
sns.despine()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='poutcome', y='y', order=['unknown', 'failure', 'other', 'success'], 
            palette=['#999999', '#e74c3c', '#3498db', '#2ecc71'], errorbar=None)

plt.title("Current Subscription Rate by Previous Outcome", pad=20)
plt.xlabel("Previous Outcome")
plt.ylabel("Subscription Rate")
plt.ylim(0, 0.7)
sns.despine()

# Add percentage labels
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}",  (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center',  xytext=(0, 5),  textcoords='offset points')
plt.show()

print("PREVIOUS CAMPAIGN OUTCOMES DISTRIBUTION:")
outcome_dist = df['poutcome'].value_counts().reset_index()
outcome_dist.columns = ['Outcome', 'Count']
outcome_dist['Percentage'] = (outcome_dist['Count'] / len(df)) * 100
print(outcome_dist.to_markdown(index=False, floatfmt=".1f"))

print("\nSUBSCRIPTION RATES BY PREVIOUS OUTCOME:")
conversion_rates = df.groupby('poutcome')['y'].agg(['count', 'mean'])
conversion_rates['mean'] = conversion_rates['mean'] * 100
conversion_rates.columns = ['Client Count', 'Subscription Rate (%)']
print(conversion_rates.sort_values('Subscription Rate (%)', ascending=False).to_markdown(floatfmt=".1f"))

#y
subscription_counts = df['y'].value_counts()
labels = ['No', 'Yes']
colors = ['#e74c3c', '#2ecc71']

# Bar plot
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=labels, y=subscription_counts.values, palette=colors)
plt.title('Term Deposit Subscription Distribution', fontsize=14, pad=20)
plt.xlabel('Subscription Status')
plt.ylabel('Count')

# percentage labels
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}\n({p.get_height()/len(df)*100:.1f}%)", 
                (p.get_x() + p.get_width()/2., p.get_height()), ha='center', 
                va='center', xytext=(0, 10), textcoords='offset points')

sns.despine()
plt.tight_layout()
plt.show()

# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(subscription_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white'})
plt.title('Subscription Ratio', fontsize=14)
plt.show()

#last question 
original_features = ['age', 'job', 'marital_status', 'education', 'credit_in_default', 'balance', 
                     'housing', 'personal_loan', 'contact', 'day', 'month', 'duration', 'campaign', 
                     'pdays', 'previous', 'poutcome', 'y']
corr_df = df_clean[original_features]

binary_cols = ['credit_in_default', 'housing', 'personal_loan', 'y']
for col in binary_cols:
    corr_df[col] = corr_df[col].map({'yes': 1, 'no': 0})

numerical_cols = corr_df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = corr_df[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Original Features)")
plt.show()