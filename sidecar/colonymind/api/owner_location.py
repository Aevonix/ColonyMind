"""The Companion phone's coordinate-free owner location contract."""
from datetime import datetime, timedelta
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from colonymind.owner_goals.hub import _companion_owner


PATH = "/v1/host/owner/location"
_COUNTRIES = frozenset("""AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO
BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ
DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF GG GH GI GL GM GN GP
GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG
KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML
MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE
PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL
SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM
US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW
""".split())


def require_companion_owner(request: Request):
    if not _companion_owner(request):
        raise HTTPException(403, detail={"code": "not_owner"})


class OwnerLocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: StrictStr
    city: StrictStr | None = Field(default=None, max_length=120)
    region: StrictStr | None = Field(default=None, max_length=60)
    country_code: StrictStr | None = None
    reported_at: StrictStr

    @field_validator("city", "region", "country_code", mode="before")
    @classmethod
    def optional_string(cls, value):
        if value is None:
            raise ValueError("omit optional strings when unknown")
        return value

    @field_validator("timezone")
    @classmethod
    def valid_zone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("timezone must be an IANA zone") from None
        return value

    @field_validator("country_code")
    @classmethod
    def valid_country(cls, value):
        if value is not None and value not in _COUNTRIES:
            raise ValueError("country_code must be an ISO 3166-1 alpha-2 code")
        return value

    @field_validator("reported_at")
    @classmethod
    def utc_stamp(cls, value):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", value):
            raise ValueError("reported_at must be a UTC ISO-8601 timestamp")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() != timedelta(0):
            raise ValueError("reported_at must be UTC")
        return parsed.isoformat().replace("+00:00", "Z")
