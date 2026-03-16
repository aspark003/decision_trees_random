import pandas as pd

df = pd.read_csv('c:/extra_tree/loader/tree.csv')
copy = df.copy()

copy.columns = copy.columns.str.replace(r'\s+',' ', regex=True).str.strip()

copy[['longitude','latitude']] = copy[['longitude','latitude']].apply(lambda x: pd.to_numeric(x, errors='coerce').astype('Float64'))

copy['ocean_proximity'] = copy['ocean_proximity'].str.replace('<1H', '', regex=True).str.replace(r'\s+',' ', regex=True).str.strip()
copy['ocean_proximity'] = pd.Categorical(copy['ocean_proximity'], categories=['NEAR BAY', 'OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'], ordered=True)
copy[['housing_median_age', 'total_rooms','total_bedrooms','population','households']] = copy[['housing_median_age', 'total_rooms','total_bedrooms','population','households']].astype('Int64')
copy[['median_income','median_house_value']] = copy[['median_income', 'median_house_value']].astype('Float64')


print(copy.head().to_string())
#copy.to_pickle('c:/extra_tree/loader/extra_tree_process.pkl')
