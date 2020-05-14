import json
from urllib.parse import quote

prefix = "http://www.semanticweb.org/eera/ontologies/2020/4/assignment4_1a#"
text_file = "JSON_RDF.txt"


def read_jsons(filename):
    with open(filename, encoding='utf-8') as json_file:
        universities = json.load(json_file)
    return universities


def add_place(uni_list, locName, countryName):
    with open(text_file, 'a',  encoding="utf-8") as txt:
        for item in uni_list:
            line = "<owl:NamedIndividual rdf:about=\"" + prefix + quote(item[locName]) \
                    + "\">\n"
            line += "<rdf:type rdf:resource=\"http://www.semanticweb.org/eera/ontologies/2020/4/" \
                    "assignment4_1a#Place\"/>\n"
            line += "<placeName>" + quote(item[locName]) + "</placeName>\n"
            line += "<countryName>" + quote(item[countryName]) + "</countryName>\n"
            line += "</owl:NamedIndividual>\n\n"
            txt.write(line)


def add_university(uni_list):
    with open(text_file, 'a',  encoding="utf-8") as txt:
        for item in uni_list:
            line = "<owl:NamedIndividual rdf:about=\"" + prefix + quote(item['universityLabel']) \
                    + "\">\n"
            line += "<rdf:type rdf:resource=\"http://www.semanticweb.org/eera/ontologies/2020/4/" \
                    "assignment4_1a#University\"/>\n"
            line += "<locatedIn rdf:resource=\"" + prefix + quote(item['location']) + "\"/>\n"
            line += "<organisationName>" + quote(item['universityLabel']) + "</organisationName>\n"
            line += "<yearFounded>" + quote(item['yearFounded']) + "</yearFounded>\n"
            line += "</owl:NamedIndividual>\n\n"
            txt.write(line)


def add_organisation(al_list):
    with open(text_file, 'a',  encoding="utf-8") as txt:
        for item in al_list:
            line = "<owl:NamedIndividual rdf:about=\"" + prefix + quote(item['organisationLabel']) \
                   + "\">\n"
            line += "<rdf:type rdf:resource=\"http://www.semanticweb.org/eera/ontologies/2020/4/" \
                    "assignment4_1a#Organisation\"/>\n"
            line += "<locatedIn rdf:resource=\"" + prefix + quote(item['orgLocationLabel']) + "\"/>\n"
            line += "<organisationName>" + quote(item['organisationLabel']) + "</organisationName>\n"
            line += "</owl:NamedIndividual>\n\n"
            txt.write(line)


def add_alumnus(al_list):
    with open(text_file, 'a',  encoding="utf-8") as txt:
        for item in al_list:
            line = "<owl:NamedIndividual rdf:about=\"" + prefix + quote(item['alumnusLabel']) \
                   + "\">\n"
            line += "<rdf:type rdf:resource=\"http://www.semanticweb.org/eera/ontologies/2020/4/" \
                    "assignment4_1a#Person\"/>\n"
            line += "<bornIn rdf:resource=\"" + prefix + quote(item['placeOfBirthLabel']) + "\"/>\n"
            line += "<alumnusOf rdf:resource=\"" + prefix + quote(item['universityLabel']) + "\"/>\n"
            line += "<employeeOf rdf:resource=\"" + prefix + quote(item['organisationLabel']) + "\"/>\n"
            line += "<personName>" + quote(item['alumnusLabel']) + "</personName>\n"
            line += "</owl:NamedIndividual>\n\n"
            txt.write(line)


universities = read_jsons('universities.json')
alumni = read_jsons('alumni2.json')
add_place(universities, 'location', 'countryLabel')
add_university(universities)
add_place(alumni, 'placeOfBirthLabel', 'pobCountryLabel')
add_place(alumni, 'orgLocationLabel', 'orgCountryLabel')
add_alumnus(alumni)


