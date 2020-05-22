
####################################   Northely  cities ###############################################
import requests
url = 'https://dbpedia.org/sparql'

query = """
SELECT DISTINCT ?citylabel (sample(?pop) as ?POP) (sample(?lat) as ?LAT)
WHERE {
   ?city rdf:type yago:City108524735 .
   ?city rdfs:label ?citylabel.
   ?city dbo:populationTotal ?pop .
   ?city geo:lat ?lat .
   FILTER ( lang(?citylabel) = 'en')
}GROUP BY ?citylabel
ORDER BY ASC(?LAT)
"""


r = requests.get(url, params = {'format': 'json', 'query': query})
data = r.json()

#Store data from DBpedia into a pandas
data['results']['bindings'][0:len(data['results']['bindings'])]
d = {'Country': [], 'Population': [] , 'Latitude': [], 'Index': []}
df = pd.DataFrame(data=d)
for i,item in enumerate(data['results']['bindings']): #for loop to store values in a df
    Country = item['citylabel']['value']
    lat = item['LAT']['value']
    population = int(item['POP']['value'])
    df = df.append({'Country': Country, 'Population':population , 'Latitude':lat , 'Index': i }, ignore_index=True)



def findCitySatisfies(cityIndex, endIndex, df):
    dfNew = df[cityIndex+1:endIndex]
    if(dfNew.empty): #reached end
        return cityIndex, 'null'
    maxValueIndex = dfNew['Population'].idxmax()
    country = dfNew['Country'][maxValueIndex]
    return maxValueIndex, country


# Recursive
cityIndex = -1 #Start with index -1 since we are adding +1 each iteration in function findCitySatisfies
endIndex = len(data['results']['bindings'])
allCities = []
allCityIndex = []

while(cityIndex != endIndex):
    cityIndexNextIter, city = findCitySatisfies(cityIndex, endIndex, df)
    if int(cityIndexNextIter) == int(cityIndex):
        break;
    else:
        allCities.append(city)
        cityIndex = cityIndexNextIter
        allCityIndex.append(cityIndex)

print(df)        
print(df.iloc[allCityIndex])





#####################   Southerly cities ##################################################


import requests
url = 'https://dbpedia.org/sparql'

query = """
SELECT DISTINCT ?citylabel (sample(?pop) as ?POP) (sample(?lat) as ?LAT)
WHERE {
   ?city rdf:type yago:City108524735 .
   ?city rdfs:label ?citylabel.
   ?city dbo:populationTotal ?pop .
   ?city geo:lat ?lat .
   FILTER ( lang(?citylabel) = 'en')
}GROUP BY ?citylabel
ORDER BY DESC(?LAT)
"""
r = requests.get(url, params = {'format': 'json', 'query': query})
data = r.json()


#Store data from DBpedia into a pandas
data['results']['bindings'][0:len(data['results']['bindings'])]
d = {'Country': [], 'Population': [] , 'Latitude': [], 'Index': []}
df = pd.DataFrame(data=d)
for i,item in enumerate(data['results']['bindings']): #for loop to store values in a df
    Country = item['citylabel']['value']
    lat = item['LAT']['value']
    population = int(item['POP']['value'])
    df = df.append({'Country': Country, 'Population':population , 'Latitude':lat , 'Index': i }, ignore_index=True)


#Function to help find city within a range
def findCitySatisfies(cityIndex, endIndex, df):
    dfNew = df[cityIndex+1:endIndex]
    if(dfNew.empty): #reached end
        return cityIndex, 'null'
    maxValueIndex = dfNew['Population'].idxmax()
    country = dfNew['Country'][maxValueIndex]
    return maxValueIndex, country


# Recursive algorithm
cityIndex = -1  #Start with index -1 since we are adding +1 each iteration in function findCitySatisfies
endIndex = len(data['results']['bindings'])
allCities = []
allCityIndex = []

while(cityIndex != endIndex):
    cityIndexNextIter, city = findCitySatisfies(cityIndex, endIndex, df)
    if int(cityIndexNextIter) == int(cityIndex):
        break;
    else:
        allCities.append(city)
        cityIndex = cityIndexNextIter
        allCityIndex.append(cityIndex)

print(df)        
print(df.iloc[allCityIndex])



