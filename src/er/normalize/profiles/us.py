from er.normalize.profiles.base import GENERIC_LEGAL, GENERIC_STREET, Profile, merged

_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia", "PR": "Puerto Rico", "GU": "Guam", "VI": "Virgin Islands",
}
REGIONS: dict[str, str] = {}
for abbr, name in _STATES.items():
    REGIONS[abbr] = f"us_{abbr.lower()}"
    REGIONS[name] = f"us_{abbr.lower()}"
REGIONS["Washington DC"] = REGIONS["D.C."] = "us_dc"

# USPS Publication 28 suffixes/units most seen in the data; canonical = short form.
_STREET = {
    "street": "st", "st": "st", "str": "st", "avenue": "ave", "ave": "ave", "av": "ave",
    "boulevard": "blvd", "blvd": "blvd", "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr",
    "court": "ct", "ct": "ct", "lane": "ln", "ln": "ln", "circle": "cir", "cir": "cir",
    "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy", "hwy": "hwy", "place": "pl", "pl": "pl",
    "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq",
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt", "building": "bldg", "bldg": "bldg",
    "floor": "fl", "fl": "fl", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt", "pt": "pt",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy", "center": "ctr", "ctr": "ctr",
}
_LEGAL_EXTRA = {"pc": "pc", "pllc": "pllc", "lc": "lc", "pa": "pa", "na": "na"}

PROFILE = Profile(
    name="US",
    legal=merged(GENERIC_LEGAL, _LEGAL_EXTRA),
    street=merged({k: v for k, v in GENERIC_STREET.items() if k not in ("nr", "near")}, _STREET),
    regions=REGIONS,
    postal_digits=5,
)
