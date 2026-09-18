from typing import Literal

from pydantic import BaseModel


class CalculateRequest(BaseModel):
    operation: Literal["growth_rate", "margin", "yoy_change", "cagr"]
    current: float | None = None
    previous: float | None = None
    numerator: float | None = None
    denominator: float | None = None
    begin_value: float | None = None
    end_value: float | None = None
    periods: float | None = None


class CalculateResponse(BaseModel):
    operation: str
    result: float | dict


class XBRLFactResponse(BaseModel):
    concept: str
    end_date: str
    value: float
    unit: str
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    filed_date: str


class XBRLResponse(BaseModel):
    cik: str
    concept: str
    facts: list[XBRLFactResponse]


class WebSearchResultResponse(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResultResponse]
