from er.normalize.profiles.base import GENERIC_LEGAL, GENERIC_STREET, Profile, merged

# canonical id → variants (English, vehicle-code abbr, old names, native-script forms seen in train data)
_STATES = {
    "in_ap": ["Andhra Pradesh", "AP", "ఆంధ్రప్రదేశ్", "ఆంధ్ర ప్రదేశ్"],
    "in_ar": ["Arunachal Pradesh", "AR"],
    "in_as": ["Assam", "AS", "অসম"],
    "in_br": ["Bihar", "BR", "बिहार"],
    "in_cg": ["Chhattisgarh", "Chattisgarh", "CG", "CT", "छत्तीसगढ़"],
    "in_ga": ["Goa", "GA"],
    "in_gj": ["Gujarat", "GJ", "ગુજરાત"],
    "in_hr": ["Haryana", "HR", "हरियाणा"],
    "in_hp": ["Himachal Pradesh", "HP", "हिमाचल प्रदेश"],
    "in_jh": ["Jharkhand", "JH", "झारखंड", "झारखण्ड"],
    "in_ka": ["Karnataka", "KA", "ಕರ್ನಾಟಕ"],
    "in_kl": ["Kerala", "Keralam", "KL", "കേരളം", "കേരള"],
    "in_mp": ["Madhya Pradesh", "MP", "मध्य प्रदेश"],
    "in_mh": ["Maharashtra", "MH", "महाराष्ट्र"],
    "in_mn": ["Manipur", "MN"],
    "in_ml": ["Meghalaya", "ML"],
    "in_mz": ["Mizoram", "MZ"],
    "in_nl": ["Nagaland", "NL"],
    "in_od": ["Odisha", "Orissa", "OD", "OR", "ଓଡ଼ିଶା", "ଓଡିଶା"],
    "in_pb": ["Punjab", "PB", "ਪੰਜਾਬ"],
    "in_rj": ["Rajasthan", "RJ", "राजस्थान"],
    "in_sk": ["Sikkim", "SK"],
    "in_tn": ["Tamil Nadu", "Tamilnadu", "TN", "தமிழ்நாடு", "தமிழ் நாடு"],
    "in_ts": ["Telangana", "TS", "TG", "తెలంగాణ"],
    "in_tr": ["Tripura", "TR"],
    "in_up": ["Uttar Pradesh", "UP", "उत्तर प्रदेश"],
    "in_uk": ["Uttarakhand", "Uttaranchal", "UK", "UA", "उत्तराखंड", "उत्तराखण्ड"],
    "in_wb": ["West Bengal", "WB", "পশ্চিমবঙ্গ"],
    "in_an": ["Andaman and Nicobar Islands", "Andaman & Nicobar Islands", "AN"],
    "in_ch": ["Chandigarh", "CH"],
    "in_dn": ["Dadra and Nagar Haveli and Daman and Diu", "Dadra and Nagar Haveli", "Daman and Diu", "DN", "DD"],
    "in_dl": ["Delhi", "DL", "NCT of Delhi", "दिल्ली"],
    "in_jk": ["Jammu and Kashmir", "Jammu & Kashmir", "JK"],
    "in_la": ["Ladakh", "LA"],
    "in_ld": ["Lakshadweep", "LD"],
    "in_py": ["Puducherry", "Pondicherry", "PY"],
}
REGIONS = {variant: cid for cid, variants in _STATES.items() for variant in variants}

_STREET = {
    "road": "rd", "rd": "rd", "street": "st", "st": "st", "marg": "marg", "nagar": "nagar", "ngr": "nagar",
    "colony": "colony", "col": "colony", "sector": "sector", "sec": "sector", "near": "near", "nr": "near",
    "opposite": "opp", "opp": "opp", "behind": "behind", "bh": "behind", "floor": "floor", "flr": "floor",
    "building": "bldg", "bldg": "bldg", "complex": "complex", "cmplx": "complex", "mohalla": "mohalla",
    "house": "no", "hno": "no", "no": "no", "number": "no", "door": "no", "plot": "plot",
    "flat": "flat", "shop": "shop", "main": "main", "cross": "cross", "layout": "layout",
    "phase": "phase", "ph": "phase", "extension": "extn", "extn": "extn", "ext": "extn",
    "first": "1st", "second": "2nd", "third": "3rd", "ground": "ground", "gf": "ground",
    "industrial": "industrial", "indl": "industrial", "estate": "estate", "apartment": "apt", "apts": "apt",
    "apartments": "apt", "apt": "apt", "post": "po", "po": "po", "village": "vill", "vill": "vill",
    "district": "dist", "dist": "dist", "taluk": "taluk", "tq": "taluk",
}
_LEGAL_EXTRA = {"opc": "opc", "pvtltd": "pvt"}

PROFILE = Profile(
    name="India",
    legal=merged(GENERIC_LEGAL, _LEGAL_EXTRA),
    street=merged(GENERIC_STREET, _STREET),
    regions=REGIONS,
    postal_digits=6,
)
