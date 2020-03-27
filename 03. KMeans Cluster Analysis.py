# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import category_encoders as ce

from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from pandas_profiling import ProfileReport
from pathlib import Path

# configure pandas to display all rows and columns without truncation
#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# %%
inFileName = 'data/01..SAMFilteredEntityData.csv'
outFileName = ''
profileFileName = 'reports/03. Profile.html'


# %%
df = pd.read_csv(inFileName, dtype={'PHYSICAL_ADDRESS_ZIP_POSTAL_CODE' : str}, na_filter=False)




# %%
df['BUSINESS_START_DATE'] = pd.to_datetime(df.BUSINESS_START_DATE, format='%Y%m%d', errors='coerce')
df['INITIAL_REGISTRATION_DATE'] = pd.to_datetime(df.INITIAL_REGISTRATION_DATE, format='%Y%m%d', errors='coerce')


# %%
# calculate the number of days since business started and business registered
df['daysSinceStart'] = (pd.datetime.now() - df['BUSINESS_START_DATE']).dt.days
df['daysSinceRegistration'] = (pd.datetime.now() - df['INITIAL_REGISTRATION_DATE']).dt.days

#%%
# drop original date columns
df = df.drop(['BUSINESS_START_DATE', 'INITIAL_REGISTRATION_DATE'], axis='columns')

# %%
# calculating daysSince* could create NaNs, so drop those records

df = df.dropna()



# %%
df['BUSINESS_TYPE_STRING'] = df['BUSINESS_TYPE_STRING'].str.split("~")
df['NAICS_CODE_STRING'] = df['NAICS_CODE_STRING'].str.split("~")
df['PSC_CODE_STRING'] = df['PSC_CODE_STRING'].str.split("~")
df['SBA_BUSINESS_TYPES_STRING'] = df['SBA_BUSINESS_TYPES_STRING'].str.split("~")


# %%
df['stateDistrict'] = df['PHYSICAL_ADDRESS_PROVINCE_OR_STATE']+'_'+df['ENTITY_CONGRESSIONAL_DISTRICT'].astype(str)
df = df.drop(['PHYSICAL_ADDRESS_PROVINCE_OR_STATE', 'ENTITY_CONGRESSIONAL_DISTRICT'], axis='columns')

# %%
# Remove NAICS codes ending in other than Y
# Y indicates that the business meets the 'small' business size standard for that particular NAICS value
# add a column to place the NAICS codes after testing for ending in Y
df['naicsY'] = df.apply(lambda x: [], axis=1)

# %%
# iterate through each row, then each element. if string contains Y, then append it to the naicsY list in that row

for i in range(len(df)):
    df.loc[i, 'naicsLen'] = len(df.loc[i,'NAICS_CODE_STRING'])
    for j in range(len(df.loc[i,'NAICS_CODE_STRING'])):
        if df.loc[i,'NAICS_CODE_STRING'][j].find("Y") > 0 :
            df.loc[i,'naicsY'].append(df.loc[i,'NAICS_CODE_STRING'][j])
    df.loc[i, 'naicsLenY'] = len(df.loc[i,'naicsY'])


# %%
# drop temporary columns
df = df.drop(['NAICS_CODE_STRING', 'naicsLen', 'naicsLenY'], axis='columns')



# %%
### ignore these two for now, figure out what to do with them later
df = df.drop(['SBA_BUSINESS_TYPES_STRING', 'FISCAL_YEAR_END_CLOSE_DATE'], axis='columns')


# %%
# df.reset_index(drop=True, inplace=True)
# profile = ProfileReport(df, title="Profile of GSA SAM Small Business Data", html={'style': {'full_width': True}})
# profile.to_file(Path(profileFileName))

# %% [markdown]
# ## Make Pipeline
# 
Column Transformer: 

* PHYSICAL_ADDRESS_ZIP_POSTAL_CODE            object ce.BinaryEncoder
* ENTITY_STRUCTURE                            object OneHotEncoder
* STATE_OF_INCORPORATION                      object ce.BinaryEncoder
* BUSINESS_TYPE_STRING                        object MultiBinarizer
* PSC_CODE_STRING                             object MultiBinarizer
* CREDIT_CARD_USAGE                           object OrdinalEncoder
* daysSinceStart                              float64 nothing
* daysSinceRegistration                       float64 nothing
* stateDistrict                               object ce.BinaryEncoder
* naicsY (was NAICS_CODE_STRING)              object MultiBinarizer


# %%
mlbCols = ['BUSINESS_TYPE_STRING', 'naicsY', 'PSC_CODE_STRING']
oheCols = ['ENTITY_STRUCTURE']
beCols = ['PHYSICAL_ADDRESS_ZIP_POSTAL_CODE', 'STATE_OF_INCORPORATION', 'stateDistrict']
oeCols = ['CREDIT_CARD_USAGE']


# %%
# can't run MultiLabelBinarizer within column transformer so have to it outside of pipeline
mlb = MultiLabelBinarizer()

# %%
dfTempBTS = pd.DataFrame(mlb.fit_transform(df['BUSINESS_TYPE_STRING']),columns=mlb.classes_)


# %%
dfTempNCS = pd.DataFrame(mlb.fit_transform(df['naicsY']),columns=mlb.classes_)

# %%
dfTempPSC = pd.DataFrame(mlb.fit_transform(df['PSC_CODE_STRING']),columns=mlb.classes_)

# %%
df = df.join(dfTempBTS)
df = df.join(dfTempNCS)
df = df.join(dfTempPSC)

# %%
# drop the mlb columns
df = df.drop(['BUSINESS_TYPE_STRING', 'naicsY', 'PSC_CODE_STRING'],  axis='columns')


# %%
# set index to the cage code value so we can trace it through to the results, but it's not used in the clustering algorithms
df = df.set_index('CAGE_CODE')

# %%
column_trans = make_column_transformer(
#   (MultiLabelBinarizer(), mlbCols),
   (OneHotEncoder(), oheCols),
   (ce.BinaryEncoder(), beCols),
   (OrdinalEncoder(), oeCols),
    remainder='passthrough'
)

# %% 
kmeans = KMeans(n_clusters=8, random_state=2020)



# %%
# pipe = Pipeline(steps=[('column_trans', column_trans),
#                         ('kmeans', kmeans)])

colTransPipe = Pipeline(steps=[('column_trans', column_trans)])

# %%
# pipe.fit_transform(df)
dfTransformed = colTransPipe.fit_transform(df)

# %%
sumSqdDistance = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(dfTransformed)
    sumSqdDistance.append(km.inertia_)
    



# %%
plt.plot(K, sumSqdDistance, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances (Inertia)')
plt.title('Elbow Method For Optimal k')
plt.show()


# %%
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
X = dfTransformed
range_n_clusters = [4, 5]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1] 
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

# %%
