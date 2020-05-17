import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SPARQLWrapper import SPARQLWrapper, JSON

OUTPUT = "output"
CACHE = f"{OUTPUT}/cache.json"

##########################################################################################
# UTILITY METHODS
##########################################################################################

def print_results(description, *results):
    print("\n==========================================================================")
    print(f"Question: {description}")
    print("==========================================================================")

    for result in results:
        print(result)

def clear_cache():
    if os.path.isfile(CACHE):
        os.remove(CACHE)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        content = json.loads(f.read())
        return content

def write_json(path, content):
    with open(path, "w+", encoding="utf-8") as f:
        json.dump(content, f, indent=4)

def query_data(endpoint, query):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results

##########################################################################################
# DATA QUERYING
##########################################################################################

ENDPOINT = "https://query.wikidata.org/sparql"

QUERY = """
    PREFIX dbpo: <http://dbpedia.org/ontology/>
    PREFIX dbpprop: <http://dbpedia.org/property/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX nobel: <http://data.nobelprize.org/terms/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?laureateName ?gender ?fieldName
                    ?birthYear ?deathYear ?awardYear
                    ?country ?countryGdpRanking
    WHERE {
        SERVICE <http://data.nobelprize.org/sparql> {
            ?laureate rdf:type nobel:Laureate.
            ?laureate rdfs:label ?laureateName.
            ?laureate foaf:gender ?gender.

            ?laureate dbpprop:dateOfBirth ?birthDate.
            BIND(year(xsd:dateTime(?birthDate)) AS ?birthYear).

            OPTIONAL {
                ?laureate dbpprop:dateOfDeath ?deathDate.
                BIND(year(xsd:dateTime(?deathDate)) AS ?deathYear).
            }
            ?laureate nobel:laureateAward ?award.
            ?award nobel:year ?awardYear.

            ?award nobel:category ?field.
            ?field rdfs:label ?fieldName.

            ?laureate dbpo:birthPlace ?birthPlace.
            ?birthPlace rdf:type dbpo:Country.
            ?birthPlace rdfs:label ?country.
            ?birthPlace (owl:sameAs|dbpo:successor) ?countryUri.

            # Work around for Netherlands (because of typo in the nobelprize dataset)
            BIND(IF(?countryUri=<http://dbpedia.org/resource/the_Netherlands>,
                    <http://dbpedia.org/resource/The_Netherlands>,
                    ?countryUri) AS ?countryDbpedia)
        }
        OPTIONAL {
            SERVICE <http://dbpedia.org/sparql> {
                {
                    ?countryDbpedia <http://dbpedia.org/property/gdpPppPerCapitaRank> ?countryGdpRanking.
                }
                UNION
                {
                    ?countryDbpedia dbpo:wikiPageRedirects ?redirectUri.
                    ?redirectUri <http://dbpedia.org/property/gdpPppPerCapitaRank> ?countryGdpRanking.
                }
            }
        }
    }
"""

def extract_dataframe(data):
    headers = data["head"]["vars"]
    results = data["results"]["bindings"]

    extract_value = lambda obj, key: obj[key]["value"] if key in obj else np.nan
    rows = [{key: extract_value(obj, key) for key in headers} for obj in results]

    df = pd.json_normalize(rows)
    df["birthYear"] = pd.to_numeric(df["birthYear"]).astype("Int32")
    df["awardYear"] = pd.to_numeric(df["awardYear"]).astype("Int32")
    df["deathYear"] = pd.to_numeric(df["deathYear"]).astype("Int32")
    df["countryGdpRanking"] = pd.to_numeric(df["countryGdpRanking"]).astype("Int32")

    print(f"Number of rows extracted: {len(rows)}")
    return df

def fetch_data():
    if os.path.isfile(CACHE):
        data = read_json(CACHE)
    else:
        data = query_data(ENDPOINT, QUERY)
        write_json(CACHE, data)
    return extract_dataframe(data)

##########################################################################################
# Question 1: What's the gender distribution among Nobel laureates?
##########################################################################################

def question1():
    description = "What's the gender distribution among Nobel laureates?"
    df = fetch_data()

    count_series = df.groupby(["gender", "fieldName"]).size()
    df = count_series.to_frame(name="size").reset_index()

    df = df.pivot(index="gender", columns="fieldName")
    df.columns = df.columns.droplevel(0)

    df["All"] = df.sum(axis=1)
    print_results(description, df.T)

    plt.figure(figsize=(16,8))
    plt.suptitle(description)

    for idx, col in enumerate(df.columns):
        ax = plt.subplot(241+idx, aspect="equal")
        df.plot(kind="pie", labels=df.index, y=col, ax=ax,
                autopct="%1.1f%%", startangle=90, legend=False, fontsize=14)

    plt.savefig(f"{OUTPUT}/question1.png")
    plt.show()

##########################################################################################
# Question 2: What's the average age of Nobel laureates at time of their awards?
##########################################################################################

def question2():
    description = "What's the average age of Nobel laureates at time of their awards?"
    df = fetch_data()

    df["age"] = df["awardYear"] - df["birthYear"]
    average_ages = df.groupby(["fieldName"])["age"].mean()
    average_ages.loc["All"] = df["age"].mean()
    print_results(description, average_ages)

    fontsize = 14
    plt.figure(figsize=(10,8))
    average_ages.plot.bar(rot=60, fontsize=fontsize)

    plt.title(description, fontsize=fontsize)
    plt.xlabel("")
    plt.ylabel("Average Age", fontsize=fontsize)
    plt.ylim((50, 70))
    plt.tight_layout()

    plt.savefig(f"{OUTPUT}/question2.png")
    plt.show()

##########################################################################################
# Question 3: Were there any Nobel laureates who were posthumously awarded?
##########################################################################################

def question3():
    description = "Were there any Nobel laureates who were posthumously awarded?"
    df = fetch_data()

    posthumously_awarded = (df["deathYear"] - df["awardYear"] <= 0)
    print_results(description, df[posthumously_awarded])

##########################################################################################
# Question 4: What's the average GDP ranking of countries where Nobel laureates were born?
##########################################################################################

def question4():
    description = "What's the average GDP ranking of countries where Nobel laureates were born?"
    df = fetch_data()
    n_total = df.shape[0]

    # Filter out all missing values from the GDP ranking
    df = df[~pd.isnull(df["countryGdpRanking"])]
    n_left = df.shape[0]

    average_ranking = df.groupby(["fieldName"])["countryGdpRanking"].mean()
    average_ranking.loc["All"] = df["countryGdpRanking"].mean()
    print_results(description, f"Result based on {n_left}/{n_total} datapoints!\n", average_ranking)

    fontsize = 14
    plt.figure(figsize=(10,8))
    average_ranking.plot.bar(rot=60, fontsize=fontsize)

    plt.title(description, fontsize=fontsize)
    plt.xlabel("")
    plt.ylabel("Average GDP Ranking", fontsize=fontsize)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT}/question4.png")
    plt.show()

##########################################################################################
# MAIN
##########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investigate data about Nobel price winners.")
    parser.add_argument("--clear-cache-before", "-cb",
                        action="store_true",
                        help="Should the cache be cleared BEFORE query execution?")
    parser.add_argument("--clear-cache-after", "-ca",
                        action="store_true",
                        help="Should the cache be cleared AFTER query execution?")
    args = parser.parse_args()

    if args.clear_cache_before:
        clear_cache()

    os.makedirs(OUTPUT, exist_ok=True)
    question1()
    question2()
    question3()
    question4()

    if args.clear_cache_after:
        clear_cache()
