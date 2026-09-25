"""Golden cases taken from real train/test record pairs."""

from er.normalize.pipeline import NormConfig, normalize_record
from er.normalize.text import fold_token

CFG = NormConfig()


def rec(name, addr="", country="US", cfg=CFG):
    return normalize_record(name, addr, country, cfg)


def test_legal_anywhere_and_dotted_initialisms():
    a = rec("L.L.C. Hines Liberty Produce")
    b = rec("llc hines liberty produce")
    c = rec("Hines Liberty Produce LLC")
    assert a["name_core"] == b["name_core"] == c["name_core"] == "hines liberty produce"
    assert a["name_legal"] == "llc"


def test_domain_matches_concat():
    assert rec("hineslibertyproduce.com")["name_concat"] == rec("Hines Liberty Produce LLC")["name_concat"]
    assert rec("fullermeese.com")["name_concat"] == rec("Fuller & Meese Inc.")["name_concat"]
    assert rec("Collgeairs.Com", country="France")["name_domain"]


def test_accent_noise_and_reorder():
    assert rec("Animal Welfare Nétwork")["name_core"] == "animal welfare network"
    assert rec("Private Qss Farmers Ltd", country="India")["name_sorted"] == \
        rec("Qss Farmers Private Limited", country="India")["name_sorted"]


def test_alias_and_id_noise():
    r = rec("Deltabelo Co aka Varsha Healthcare Limited", country="India")
    assert r["name_alts"] == ["deltabelo", "healthcare varsha"]
    assert rec("OB Point Private (ID: 45404)", country="India")["name_core"] == "ob point"
    assert rec("OB OB Point Private", country="India")["name_core"] == "ob point"


def test_regions_all_forms():
    for addr in ("Chennai, Tamil Nadu", "Chennai, TN", "CHENNAI, தமிழ்நாடு", "Chennai Tamil Nadu 600001"):
        assert rec("x", addr, "India")["addr_region"] == "in_tn", addr
    assert rec("x", "1361 Norby Rd, Jamestown, New York")["addr_region"] == "us_ny"
    assert rec("x", "1361 NORBY ROAD, KIANTONE, NY")["addr_region"] == "us_ny"
    assert rec("x", "54 RUE X, TOURCOING, Nord", "France")["addr_region"] == "fr_hdf"
    assert rec("x", "23 Rue Y, La Teste-de-buch, Gironde", "France")["addr_region"] == \
        rec("x", "23 Rue Y, La Teste-de-Buch, Nouvelle-Aquitaine", "France")["addr_region"]


def test_street_canon_and_house():
    a = rec("x", "1286 1/2 KENTS RIDGE RD, RICHLANDS TOWN, VA")
    b = rec("x", "1286 Kents Ridge Road, Richlands Town, Virginia")
    assert a["addr_house"] == b["addr_house"] == "1286"
    assert a["addr_content"] == b["addr_content"] == "kents ridge richlands town"
    assert rec("x", "MN, Belle Plaine, 24221 Kittson Boulevard")["addr_house"] == "24221"
    assert rec("x", "MN, Belle Plaine, 24221 Kittson Boulevard")["addr_postal"] == ""
    assert rec("x", "Flat No-D-107, Panda Residency, Bhubaneswar", "India")["addr_house"] == "107"
    assert rec("x", "No ##55 (125/2), TN, Chennai", "India")["addr_house"] == "55"
    fr = rec("x", "Nº 9 R LOUIS SPRIET, LILLE, Hauts-de-France", "France")
    assert fr["addr_house"] == "9" and "9 rue louis spriet lille fr_hdf" in fr["addr_norm"]
    assert rec("x", "12 Rue X, ST.-NAZAIRE", "France")["addr_norm"] == "12 rue x saint nazaire"


def test_postal():
    assert rec("x", "797, Lake Town, Kolkata, West Bengal 700089", "India")["addr_postal"] == "700089"
    assert rec("x", "1795 Westchester Drive, High Point, NC 27260")["addr_postal"] == "27260"
    assert rec("x", "10 Rue X, 59000 LILLE", "France")["addr_postal"] == "59000"


def test_open_set_country():
    r = rec("Acme GmbH", "Hauptstrasse 5, Berlin", "Germany")
    assert r["profile"] == "GENERIC" and r["name_core"] == "acme"
    assert rec("x", "Chennai, TN", "India", NormConfig(profile_mode="generic"))["addr_region"] == ""


def test_city_unit_landmark():
    assert rec("x", "1361 Norby Rd, Jamestown, New York")["addr_city"] == "jamestown"
    assert rec("x", "797, Lake Town, Kolkata, West Bengal 700089", "India")["addr_city"] == "kolkata"
    assert rec("x", "10 Rue X, 59000 LILLE", "France")["addr_city"] == "lille"
    assert rec("x", "123 Main St Apt 4B, Springfield, IL")["addr_unit"] == "4b"
    assert rec("x", "123 Main St, Springfield, IL")["addr_unit"] == ""
    assert rec("x", "Near Big Bazaar, MG Road, Bangalore", "India")["addr_landmark"] == "big bazaar"
    assert rec("x", "MG Road, Bangalore", "India")["addr_landmark"] == ""
    # additive: addr_content/addr_house/addr_region/addr_postal stay exactly as before (Step 2/3 parity)
    a = rec("x", "1286 1/2 KENTS RIDGE RD, RICHLANDS TOWN, VA")
    assert a["addr_content"] == "kents ridge richlands town"
    assert a["addr_house"] == "1286"


def test_native_translit_fold():
    r = rec("आनंद कंसल्टेंट्स प्राइवेट लिमिटेड", country="India")
    assert r["name_native"] and "ltd" in r["name_legal"]  # praivet→pvt comes from learned N4 map
    assert fold_token("consultants") == fold_token("kmsltemts")
    assert fold_token("vijay") == fold_token("bijy")
    assert fold_token("private") == fold_token("praivet")
