"""Open-set profile registry: unknown country labels fall back to GENERIC, never filtered."""

from er.normalize.profiles import france, india, us
from er.normalize.profiles.base import GENERIC_LEGAL, GENERIC_STREET, Profile

GENERIC = Profile(name="GENERIC", legal=GENERIC_LEGAL, street=GENERIC_STREET, regions={})

_BY_LABEL = {
    "us": us.PROFILE, "usa": us.PROFILE, "united states": us.PROFILE, "united states of america": us.PROFILE,
    "india": india.PROFILE, "in": india.PROFILE, "bharat": india.PROFILE,
    "france": france.PROFILE, "fr": france.PROFILE, "fra": france.PROFILE,
}
ALL = (us.PROFILE, india.PROFILE, france.PROFILE, GENERIC)


def profile_for(country: str | None, mode: str = "auto") -> Profile:
    if mode == "generic":
        return GENERIC
    return _BY_LABEL.get((country or "").strip().lower(), GENERIC)
