import json

prefix = "http://www.semanticweb.org/eera/ontologies/2020/4/assignment4_1a#"
text_file = "JSON_RDF.txt"


def read_jsons(filename):
    with open(filename, encoding='utf-8') as json_file:
        universities = json.load(json_file)
    return universities


def add_place(uni_list, locName, countryName):
    with open(text_file, 'a',  encoding="utf-8") as txt:
        for item in uni_list:
            line = "<owl:NamedIndividual rdf:about=\"" + prefix + item[locName].replace(' ', '_') \
                    + "\">\n"
            line += "<rdf:type rdf:resource=\"http://www.semanticweb.org/eera/ontologies/2020/4/" \
                    "assignment4_1a#Place\"/>\n"
            line += "<placeName>" + item[locName].replace(' ', '_') + "</placeName>\n"
            line += "<countryName>" + item[countryName].replace(' ', '_') + "</countryName>\n"
            line += "</owl:NamedIndividual>\n\n"
            txt.write(line)


def add_university(uni_list):
    with open(text_file, 'a',  encoding="utf-8") as txt:
        for item in uni_list:
            line = "<owl:NamedIndividual rdf:about=\"" + prefix + item['universityLabel'].replace(' ', '_') \
                    + "\">\n"
            line += "<rdf:type rdf:resource=\"http://www.semanticweb.org/eera/ontologies/2020/4/" \
                    "assignment4_1a#University\"/>\n"
            line += "<locatedIn rdf:resource=\"" + prefix + item['location'].replace(' ', '_') + "\"/>\n"
            line += "<organisationName>" + item['universityLabel'].replace(' ', '_') + "</organisationName>\n"
            line += "<yearFounded>" + item['yearFounded'] + "</yearFounded>\n"
            line += "</owl:NamedIndividual>\n\n"
            txt.write(line)


def add_organisation(al_list):
    with open(text_file, 'a',  encoding="utf-8") as txt:
        for item in al_list:
            if 'University' in item['organisationLabel'] or 'Institute' in item['organisationLabel']:
                line = "<owl:NamedIndividual rdf:about=\"" + prefix + item['organisationLabel'].replace(' ', '_') \
                       + "\">\n"
                line += "<rdf:type rdf:resource=\"http://www.semanticweb.org/eera/ontologies/2020/4/" \
                        "assignment4_1a#University\"/>\n"
                line += "<locatedIn rdf:resource=\"" + prefix + item['orgLocationLabel'].replace(' ', '_') + "\"/>\n"
                line += "<organisationName>" + item['organisationLabel'].replace(' ', '_') + "</organisationName>\n"
                if 'yearFounded' in item:
                    line += "<yearFounded>" + item['yearFounded'] + "</yearFounded>\n"
                line += "</owl:NamedIndividual>\n\n"
                txt.write(line)
            else:
                line = "<owl:NamedIndividual rdf:about=\"" + prefix + item['organisationLabel'].replace(' ', '_') \
                       + "\">\n"
                line += "<rdf:type rdf:resource=\"http://www.semanticweb.org/eera/ontologies/2020/4/" \
                        "assignment4_1a#Organisation\"/>\n"
                line += "<locatedIn rdf:resource=\"" + prefix + item['orgLocationLabel'].replace(' ', '_') + "\"/>\n"
                line += "<organisationName>" + item['organisationLabel'].replace(' ', '_') + "</organisationName>\n"
                line += "</owl:NamedIndividual>\n\n"
                txt.write(line)


def add_alumnus(al_list):
    with open(text_file, 'a',  encoding="utf-8") as txt:
        for item in al_list:
            line = "<owl:NamedIndividual rdf:about=\"" + prefix + item['alumnusLabel'].replace(' ', '_') \
                   + "\">\n"
            line += "<rdf:type rdf:resource=\"http://www.semanticweb.org/eera/ontologies/2020/4/" \
                    "assignment4_1a#Person\"/>\n"
            line += "<bornIn rdf:resource=\"" + prefix + item['placeOfBirthLabel'].replace(' ', '_') + "\"/>\n"
            line += "<alumnusOf rdf:resource=\"" + prefix + item['universityLabel'].replace(' ', '_') + "\"/>\n"
            line += "<employeeOf rdf:resource=\"" + prefix + item['organisationLabel'].replace(' ', '_') + "\"/>\n"
            line += "<personName>" + item['alumnusLabel'].replace(' ', '_') + "</personName>\n"
            line += "</owl:NamedIndividual>\n\n"
            txt.write(line)


universities = read_jsons('universities.json')
alumni = read_jsons('alumni2.json')
add_place(universities, 'location', 'countryLabel')
add_university(universities)
add_place(alumni, 'placeOfBirthLabel', 'pobCountryLabel')
add_place(alumni, 'orgLocationLabel', 'orgCountryLabel')
add_organisation(alumni)
add_alumnus(alumni)


