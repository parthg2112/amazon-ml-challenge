"""France profile: hand-defined; France has no training labels (see PLAN.md, leave-profile-out)."""

from er.normalize.profiles.base import GENERIC_LEGAL, Profile, merged

# region id → region names (current + pre-2016) and its départements
_REGIONS = {
    "fr_ara": ["Auvergne-Rhône-Alpes", "ARA", "Auvergne", "Rhône-Alpes", "Ain", "Allier", "Ardèche", "Cantal",
               "Drôme", "Isère", "Loire", "Haute-Loire", "Puy-de-Dôme", "Rhône", "Savoie", "Haute-Savoie"],
    "fr_bfc": ["Bourgogne-Franche-Comté", "BFC", "Bourgogne", "Franche-Comté", "Côte-d'Or", "Doubs", "Jura",
               "Nièvre", "Haute-Saône", "Saône-et-Loire", "Yonne", "Territoire de Belfort"],
    "fr_bre": ["Bretagne", "Côtes-d'Armor", "Finistère", "Ille-et-Vilaine", "Morbihan"],
    "fr_cvl": ["Centre-Val de Loire", "CVL", "Centre", "Cher", "Eure-et-Loir", "Indre", "Indre-et-Loire",
               "Loir-et-Cher", "Loiret"],
    "fr_cor": ["Corse", "Corse-du-Sud", "Haute-Corse"],
    "fr_ges": ["Grand Est", "Alsace", "Lorraine", "Champagne-Ardenne", "Ardennes", "Aube", "Marne",
               "Haute-Marne", "Meurthe-et-Moselle", "Meuse", "Moselle", "Bas-Rhin", "Haut-Rhin", "Vosges"],
    "fr_hdf": ["Hauts-de-France", "HDF", "Nord-Pas-de-Calais", "Picardie", "Aisne", "Nord", "Oise",
               "Pas-de-Calais", "Somme"],
    "fr_idf": ["Île-de-France", "IDF", "Paris", "Seine-et-Marne", "Yvelines", "Essonne", "Hauts-de-Seine",
               "Seine-Saint-Denis", "Val-de-Marne", "Val-d'Oise"],
    "fr_nor": ["Normandie", "Basse-Normandie", "Haute-Normandie", "Calvados", "Eure", "Manche", "Orne",
               "Seine-Maritime"],
    "fr_naq": ["Nouvelle-Aquitaine", "NAQ", "Aquitaine", "Limousin", "Poitou-Charentes", "Charente",
               "Charente-Maritime", "Corrèze", "Creuse", "Dordogne", "Gironde", "Landes", "Lot-et-Garonne",
               "Pyrénées-Atlantiques", "Deux-Sèvres", "Vienne", "Haute-Vienne"],
    "fr_occ": ["Occitanie", "OCC", "Midi-Pyrénées", "Languedoc-Roussillon", "Ariège", "Aude", "Aveyron",
               "Gard", "Haute-Garonne", "Gers", "Hérault", "Lot", "Lozère", "Hautes-Pyrénées",
               "Pyrénées-Orientales", "Tarn", "Tarn-et-Garonne"],
    "fr_pdl": ["Pays de la Loire", "PDL", "Loire-Atlantique", "Maine-et-Loire", "Mayenne", "Sarthe", "Vendée"],
    "fr_pac": ["Provence-Alpes-Côte d'Azur", "PACA", "Alpes-de-Haute-Provence", "Hautes-Alpes",
               "Alpes-Maritimes", "Bouches-du-Rhône", "Var", "Vaucluse"],
    "fr_dom": ["Guadeloupe", "Martinique", "Guyane", "La Réunion", "Réunion", "Mayotte"],
}
REGIONS = {variant: rid for rid, variants in _REGIONS.items() for variant in variants}

_STREET = {
    "rue": "rue", "r": "rue", "avenue": "av", "av": "av", "ave": "av", "boulevard": "bd", "bd": "bd",
    "blvd": "bd", "bld": "bd", "chemin": "ch", "ch": "ch", "chem": "ch", "impasse": "imp", "imp": "imp",
    "allee": "all", "all": "all", "place": "pl", "pl": "pl", "route": "rte", "rte": "rte", "quai": "qu",
    "qu": "qu", "square": "sq", "sq": "sq", "cours": "crs", "crs": "crs", "cour": "crs",
    "faubourg": "fbg", "fbg": "fbg", "residence": "res", "res": "res", "lieu": "ld", "ld": "ld",
    "saint": "saint", "st": "saint", "sainte": "sainte", "ste": "sainte", "numero": "no", "no": "no",
    "n": "no", "bis": "bis", "ter": "ter", "etage": "etage", "batiment": "bat", "bat": "bat",
    "zone": "za", "za": "za", "zi": "zi", "zac": "zac", "cedex": "cedex", "bp": "bp",
}
_LEGAL_EXTRA = {
    "sa": "sa", "sci": "sci", "snc": "snc", "scop": "scop", "sca": "sca", "selarl": "selarl",
    "cie": "cie", "ets": "ets", "etablissements": "ets", "scs": "scs", "gie": "gie", "sasu": "sasu",
}

PROFILE = Profile(
    name="France",
    legal=merged(GENERIC_LEGAL, _LEGAL_EXTRA),
    street=_STREET,
    regions=REGIONS,
    postal_digits=5,
    postal_before_city=True,
    apostrophe_splits=True,
    extra_stopwords=frozenset({"a", "au", "aux", "en", "sur"}),
    # "saint" is a street-name word (Rue Saint-Joseph), not a street type; kept in _STREET so ST.- variants
    # canonicalize to saint.
    street_content_keep=frozenset({"saint", "sainte"}),
)
