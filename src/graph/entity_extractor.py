import spacy
nlp = spacy.load('en_core_web_sm')

def extract_entities_and_relations(text: str):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({'text': ent.text, 'label': ent.label_})
    # simple relation extraction: (head -> child) pairs for named entities
    relations = []
    for sent in doc.sents:
        ents = [ent for ent in sent.ents]
        # pairwise relations inside sentence via dependency
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                relations.append({'source': ents[i].text, 'target': ents[j].text, 'type': 'co-occur'})
    return entities, relations